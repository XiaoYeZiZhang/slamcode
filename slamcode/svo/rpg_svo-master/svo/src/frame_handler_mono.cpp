// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
#include <svo/sparse_img_align.h>
#include <vikit/performance_monitor.h>
#include <svo/depth_filter.h>
#ifdef USE_BUNDLE_ADJUSTMENT
#include <svo/bundle_adjustment.h>
#endif

namespace svo {

FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera* cam) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(NULL)
{
  initialize();
}


// 特征检测 深度滤波
void FrameHandlerMono::initialize()
{
  feature_detection::DetectorPtr feature_detector(
      new feature_detection::FastDetector(
          cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));
          	
  DepthFilter::callback_t depth_filter_cb = boost::bind(
      &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
  // 这里第二个参数是一个callback_t类型的函数，也就是上面一行
  depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb);
  // 创建深度滤波器线程  调用depth_filter.cpp中的startThread方法
  depth_filter_->startThread();
}

FrameHandlerMono::~FrameHandlerMono()
{
  delete depth_filter_;
}


//此函数应该是不断被调用   连续处理进来的图像
void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp)
{
  if(!startFrameProcessingCommon(timestamp))
    return;

  // some cleanup from last iteration, can't do before because of visualization
  core_kfs_.clear();
  overlap_kfs_.clear();

  // create new frame
  // 形成新的当前帧
  SVO_START_TIMER("pyramid_creation");
  // 新帧的初始化  构建图像金字塔
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  SVO_STOP_TIMER("pyramid_creation");

  // process frame
  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
    res = processFrame();
  // 处理第二帧图像
  else if(stage_ == STAGE_SECOND_FRAME)
    res = processSecondFrame();
  else if(stage_ == STAGE_FIRST_FRAME)
    //处理第一帧图像  主要是提取了角点信息
    res = processFirstFrame();
  else if(stage_ == STAGE_RELOCALIZING)
    // 单位R和零t  和当前帧有共视点的临近关键帧
    res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
                          map_.getClosestKeyframe(last_frame_));

  // set last frame
  // 将上一帧设置为当前帧
  last_frame_ = new_frame_;
  
  //.reset()函数将set_reset_ = true;
  // new_frame_是FramePtr类型，typedef boost::shared_ptr<Frame> FramePtr;
  //?????????????????????????????????没找到这个函数
  new_frame_.reset();
  
  // finish processing
  // res是帧处理后的返回状态
  // nObs()定义在frame.h中 返回特征点的数目
  // 根据帧处理的返回状态进行下一步的状态处理
  // 在frame_handler_base.cpp中
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
}

// 处理第一帧  提取角点
FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
  //将第一帧的位姿初始化：旋转矩阵为单位矩阵，零位移
  new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
  // initialization::KltHomographyInit klt_homography_init_;
  // 提取第一张图像的角点，得到角点的像素坐标和在相机参考系下的单位球坐标
  // 如果提取的角点数小于100，返回FAILURE
  if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
    return RESULT_NO_KEYFRAME;

  // 将第一帧设置为关键帧  寻找分较开的关键点 放在key_pts_中
  new_frame_->setKeyframe();
  // 将该关键帧 添加到map地图中
  map_.addKeyframe(new_frame_);
  // 开始处理第二帧
  stage_ = STAGE_SECOND_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");
  return RESULT_IS_KEYFRAME;
}

/*处理第二帧  得到单应矩阵   三角化得到在当前帧的三维坐标   scale  形成地图点(相对于第一帧)  将地图点与相关联的帧建立联系
此时每一帧的位姿都是相对于第一帧  而且是scale后的
将当前帧与前一个关键帧进行lobalBA  优化前一个关键帧的位姿和地图点  删除误差大于阈值的地图点
设置当前帧的关键角点   初始化深度滤波器   将当前帧作为关键帧插入地图*/
FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
	// initialization.cpp
  // 处理第二帧  得到单应矩阵   三角化得到在当前帧的三维坐标   scale  形成地图点(相对于第一帧)  将地图点与相关联的帧建立联系
  // 此时每一帧的位姿都是相对于第一帧  而且是scale后的
  initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_);
  
  if(res == initialization::FAILURE)
    return RESULT_FAILURE;
  else if(res == initialization::NO_KEYFRAME)
    // 视差太小
    return RESULT_NO_KEYFRAME;

  // two-frame bundle adjustment
  // 当前帧和前一个关键帧 进行ba
#ifdef USE_BUNDLE_ADJUSTMENT
  // bundle_adjustment.cp中
  // Config::lobaThresh():2.0
  // 对当前帧和前一个关键帧  进行localBA，优化前一个关键帧的位姿和所有的地图点位置
  // 对于优化边误差不满足阈值的，将对应的地图点删除
  ba::twoViewBA(new_frame_.get(), map_.lastKeyframe().get(), Config::lobaThresh(), &map_);
#endif

  // 对于当前帧 设置五个关键特征点
  new_frame_->setKeyframe();
  double depth_mean, depth_min;
  // frame.cpp 174
  // 得到在当前帧坐标系下关联地图点的平均深度和最小深度
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  // ???为当前帧的每个特征点初始化一个深度种子
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  // add frame to map
  // 将当前帧作为关键帧  加入地图
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  klt_homography_init_.reset();
  SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");
  return RESULT_IS_KEYFRAME;
}



// 处理除了第一帧和第二帧图像
FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
  // Set initial pose TODO use prior
  // 新帧的位姿初始值设置为上一帧的位姿(相对于世界坐标系)
  new_frame_->T_f_w_ = last_frame_->T_f_w_;

  // sparse image align
  SVO_START_TIMER("sparse_img_align");
  
  // 4 2 30(迭代次数) 进行初始化
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
  
  // 利用上一帧  当前帧 优化相对位姿 更新当前帧相对于世界坐标系的位姿
  size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
  SVO_STOP_TIMER("sparse_img_align");
  SVO_LOG(img_align_n_tracked);
  SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);


  // 得到了当前帧位姿的粗略估计  是基于上一帧的结果计算的，所以有较大的累积误
  // 需要进一步和地图之间进行特征比对来对当前位姿进行进一步优化
  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  // reprojector.cpp中  利用其它关键帧关联的地图点  使用光流优化在当前帧中的像素位置，形成当前帧的特征点
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  
  // 在当前帧中新生成的特征点数量
  const size_t repr_n_new_references = reprojector_.n_matches_;
  // 找到n_matches_所需要的cell的数量
  const size_t repr_n_mps = reprojector_.n_trials_;
  SVO_LOG2(repr_n_mps, repr_n_new_references);
  SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);
  
  // 50
  // 如果没有找到50个成功投影到当前帧的地图点，则将当前帧的位姿设置为前一帧的位姿
  if(repr_n_new_references < Config::qualityMinFts())
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    tracking_quality_ = TRACKING_INSUFFICIENT;
    return RESULT_FAILURE;
  }


  // pose optimization
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  
  // pose_optimizer.cpp
  // 利用重投影误差 优化位姿  最后根据位姿  得到那些在误差范围的当前帧的角点 对应的point不为空
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
      
  SVO_STOP_TIMER("pose_optimizer");
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit =  "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
  // sfba_n_edges_final是当前帧的角点中 满足重投影误差误差范围的角点数量
  if(sfba_n_edges_final < 20)
    return RESULT_FAILURE;



  // structure optimization
  SVO_START_TIMER("point_optimizer");
  // 当前帧  20  5
  // frame_handler_base.cpp  根据重投影误差求地图点的导数  优化地图点的三维坐标
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  SVO_STOP_TIMER("point_optimizer");

  // select keyframe
  core_kfs_.insert(new_frame_);
  // sfba_n_edges_final是当前帧中 根据优化后的位姿  满足重投影误差范围的角点数量
  // frame_handler_base.cpp中  对于角点的数量 利用阈值进行判断  输出信息
  setTrackingQuality(sfba_n_edges_final);
  
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    return RESULT_FAILURE;
  }
  
  
  
  
  double depth_mean, depth_min;
  // frame.cpp  获得地图点在当前帧坐标系下的平均深度和最小深度
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);

  // 判断是否将当前帧作为关键帧
  // 如果当前帧不是关键帧  则用来收敛深度
  if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
  {
  	// 利用当前帧 更新地图seed的深度
    depth_filter_->addFrame(new_frame_);
    return RESULT_NO_KEYFRAME;
  }
  
  
  // 将当前帧设置为关键帧
  new_frame_->setKeyframe();
  SVO_DEBUG_STREAM("New keyframe selected.");

  // new keyframe selected
  // 遍历当前帧的匹配角点
  // ?????????????????????????????当前帧的fts_是由前一帧光流得来的么????????????????
  for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    if((*it)->point != NULL)
    	// 建立地图点和当前帧的联系
      (*it)->point->addFrameRef(*it);
      
  // candidate_怎么初始化?????????????
  // 怎么产生新的地图点??????????????????addKeyframe
  map_.point_candidates_.addCandidatePointToFrame(new_frame_);

  // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  // 0
  if(Config::lobaNumIter() > 0)
  {
    SVO_START_TIMER("local_ba");
    setCoreKfs(Config::coreNKfs());
    size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;
    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                loba_n_erredges_init, loba_n_erredges_fin,
                loba_err_init, loba_err_fin);
    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                     "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
  }
#endif

  // init new depth-filters
  // 进行当前帧的角点检测  利用当前帧的平均深度和最小深度初始化种子
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  // if limited number of keyframes, remove the one furthest apart
  // 如果关键帧数目太多  删除最早的关键帧
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
  	// 根据当前关键帧的位置  找打离得最远的关键帧furthest_frame
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    
    // 将由它产生的非收敛种子点删除
    depth_filter_->removeKeyframe(furthest_frame); // TODO this interrupts the mapper thread, maybe we can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

  // add keyframe to map
  // 将当前关键帧添加进地图
  map_.addKeyframe(new_frame_);

  return RESULT_IS_KEYFRAME;
}


// 对于跟丢的帧 进行重定位  用有共视点的关键帧 进行重定位
// 单位R和零t  和当前帧有共视点的临近关键帧
// 这里什么时候进行重定位呢????????????????
FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(
    const SE3& T_cur_ref,
    FramePtr ref_keyframe)
{
  SVO_WARN_STREAM_THROTTLE(1.0, "Relocalizing frame");
  if(ref_keyframe == nullptr)
  {
    SVO_INFO_STREAM("No reference keyframe.");
    return RESULT_FAILURE;
  }
  

  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
  // 根据重投影误差 优化参考帧到当前帧的相对位姿
  // ?????????????????如果当前帧跟丢了 当前帧的位姿 是上一帧的位姿么   是的
  // ?????????????????img_align_n_tracked这里指的什么????参与计算的角点数量????????/
  // 得到优化后的当前位姿  不要了?????????????
  size_t img_align_n_tracked = img_align.run(ref_keyframe, new_frame_);
  
  
  if(img_align_n_tracked > 30)
  {
  	// 得到上一帧的位姿
    SE3 T_f_w_last = last_frame_->T_f_w_;
    // 将上一帧设置为有公式点的关键帧
    last_frame_ = ref_keyframe;
    // 重新进行帧的处理   将当前帧的位姿初始化为临近关键帧的位姿????????
    // 所以上面的align步骤得到的当前帧位姿没有用???????????????
    FrameHandlerMono::UpdateResult res = processFrame();
    if(res != RESULT_FAILURE)
    {
      stage_ = STAGE_DEFAULT_FRAME;
      SVO_INFO_STREAM("Relocalization successful.");
    }
    else
    	// 如果重定位失败，将当前帧位姿设置为上一帧的位姿?????????
      new_frame_->T_f_w_ = T_f_w_last; // reset to last well localized pose
    return res;
  }
  return RESULT_FAILURE;
}



bool FrameHandlerMono::relocalizeFrameAtPose(
    const int keyframe_id,
    const SE3& T_f_kf,
    const cv::Mat& img,
    const double timestamp)
{
  FramePtr ref_keyframe;
  if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
    return false;
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
  if(res != RESULT_FAILURE) {
    last_frame_ = new_frame_;
    return true;
  }
  return false;
}

// 重置发生在处理第一帧和第二帧 跟踪失败的时候  需要重新来过
void FrameHandlerMono::resetAll()
{
  resetCommon();
  last_frame_.reset();
  new_frame_.reset();
  core_kfs_.clear();
  overlap_kfs_.clear();
  depth_filter_->reset();
}

void FrameHandlerMono::setFirstFrame(const FramePtr& first_frame)
{
  resetAll();
  last_frame_ = first_frame;
  last_frame_->setKeyframe();
  map_.addKeyframe(last_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
}


// 根据当前帧角点的平均深度  判断是否作为关键帧插入地图
bool FrameHandlerMono::needNewKf(double scene_depth_mean)
{
	// 遍历当前帧的所有共视关键帧
  for(auto it=overlap_kfs_.begin(), ite=overlap_kfs_.end(); it!=ite; ++it)
  {
  	// 共视关键帧在第一帧中的位置
  	// w2f 将共视关键帧的位置乘以当前帧的位姿
  	// 得到共视关键帧 在当前帧中的位置??????????????????????
  	// frame.h  
    Vector3d relpos = new_frame_->w2f(it->first->pos());
    
    if(fabs(relpos.x())/scene_depth_mean < Config::kfSelectMinDist() &&
       fabs(relpos.y())/scene_depth_mean < Config::kfSelectMinDist()*0.8 &&
       fabs(relpos.z())/scene_depth_mean < Config::kfSelectMinDist()*1.3)
      return false;
  }
  return true;
}

void FrameHandlerMono::setCoreKfs(size_t n_closest)
{
  size_t n = min(n_closest, overlap_kfs_.size()-1);
  std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin()+n, overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                    boost::bind(&pair<FramePtr, size_t>::second, _2));
  std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i){ core_kfs_.insert(i.first); });
}

} // namespace svo
