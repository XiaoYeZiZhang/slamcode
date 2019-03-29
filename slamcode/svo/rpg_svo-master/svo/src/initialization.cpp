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
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/initialization.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>

namespace svo {
namespace initialization {

InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
  reset();
  // 参考帧  参考帧的关键点   当前帧关键点  就在initialization.cpp中
  // 对于第一帧图像  px_ref_存放的是角点的像素坐标  f_ref_存放的是角点在相机坐标系下的单位球坐标  Z取1然后归一化
  detectFeatures(frame_ref, px_ref_, f_ref_);

  // 如果第一张图像检测的角点小于100个，失败，直接返回
  if(px_ref_.size() < 100)
  {
    SVO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
    return FAILURE;
  }

  // 将第一帧当做参考帧
  frame_ref_ = frame_ref;
  // px_cur_存放图像角点的像素坐标
  // 将第一帧图像角点的像素坐标插入 px_cur_中
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());
  return SUCCESS;
}


// 添加第二帧图像
InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
  // 第一帧 当前帧  第一帧的角点像素坐标 待求，会返回第二帧匹配的像素坐标   第一帧角点在相机坐标系下的归一化坐标
  // 光流法  得到第二帧图像对应的角点坐标px_cur_，单位球坐标f_cur_  和与第一帧对应角点的距离disparities_
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  SVO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");
  

  // 如果匹配的点小于50 则失败
  if(disparities_.size() < Config::initMinTracked())
    return FAILURE;

  // 得到所有匹配角点视差的平均值
  double disparity = vk::getMedian(disparities_);
  
  SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");
  // 50.0
  // 如果视差过小，则不把当前帧视为关键帧
  if(disparity < Config::initMinDisparity())
    return NO_KEYFRAME;

  // Config::poseOptimThresh()：2.0
  // 计算单应矩阵T_cur_from_ref_  三角化得到三维坐标xyz_in_cur_ 得到匹配角点的内点inliers_
  computeHomography(
      f_ref_, f_cur_,
      frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
      inliers_, xyz_in_cur_, T_cur_from_ref_);
      
      
  SVO_INFO_STREAM("Init: Homography RANSAC "<<inliers_.size()<<" inliers.");

  // 如果匹配点的内点小于40 返回失败
  if(inliers_.size() < Config::initMinInliers())
  {
    SVO_WARN_STREAM("Init WARNING: "<<Config::initMinInliers()<<" inliers minimum required.");
    return FAILURE;
  }



  // Rescale the map such that the mean scene depth is equal to the specified scale
  // depth_vec存储三角化得到三维点的深度Z
  vector<double> depth_vec;
  // 遍历所有三角化产生的三维点
  for(size_t i=0; i<xyz_in_cur_.size(); ++i)
    depth_vec.push_back((xyz_in_cur_[i]).z());
  // 得到平均深度
  double scene_depth_median = vk::getMedian(depth_vec);
  // mapScale()为1.0  scale为平均深度的倒数
  double scale = Config::mapScale()/scene_depth_median;

  // 得到从第一帧到当前帧的位姿
  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;
  // pos是该帧在第一帧坐标系下的位置
  // 利用尺度变换了t
  // ??????????????????????????? 归一化深度是怎么计算的??????????????????????
  frame_cur->T_f_w_.translation() =
      -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()));

  // For each inlier create 3D point and add feature in both frames
  // scale后的从当前帧到第一帧的位姿
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();

  for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
  {
    // 得到属于内点的像素坐标
    Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
    Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);

    // 如果当前帧对应的内点在参考帧边缘10以内  参考帧对应的内点在当前帧边缘10以内， 三角化的三维点深度为正
    if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
    {
      // 转换到第一帧对应的三维坐标  三维坐标进行了尺度变换
      Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);
      Point* new_point = new Point(pos);



      // 从帧可以得到角点对应的三维坐标(相对于第一帧)， 从地图点可以得到看到它的帧 和对应的像素坐标  求坐标系下的坐标
      ////  当前帧 在第一帧下的scale后的三维点  在当前帧的像素坐标   当前帧的球坐标系下坐标
      Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
      //保存在帧的fts_中
      frame_cur->addFeature(ftr_cur);
      //保存在地图点的obs_中 因为由此feature可以得到对应的帧
      new_point->addFrameRef(ftr_cur);

      // //// 参考帧 在第一帧下的scale后的三维点  在参考帧的像素坐标   参考帧的球坐标系下坐标
      Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
      frame_ref_->addFeature(ftr_ref);
      new_point->addFrameRef(ftr_ref);
    }
  }
  return SUCCESS;
}

void KltHomographyInit::reset()
{
  px_cur_.clear();
  frame_ref_.reset();
}


// 提取特征
// 参考帧  参考帧的关键点  当前帧的关键点
void detectFeatures(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec)
{
  Features new_features;
  // gridSize: 30  nPyrLevels：3 在config.h中定义
  feature_detection::FastDetector detector(
      frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());

  // 得到角点的像素坐标 new_features
  // 在feature_detection.cpp中
  detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);

  // now for all maximum corners, initialize a new seed
  // px_vec存储角点的像素坐标
  // f_vec存储角点在相机坐标系下的单位球坐标(X/Z,Y/Z,1)归一化
  px_vec.clear(); px_vec.reserve(new_features.size());
  f_vec.clear(); f_vec.reserve(new_features.size());

  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
    // 这里feature的f函数 在feature.h中，是转换为相机坐标系下的单位球坐标????????????????????????????????????
    f_vec.push_back(ftr->f);
    delete ftr;
  });
}


/*
第一帧
当前帧  对于每一帧  已经提取了图像金字塔
第一帧的角点像素坐标   
待求当前帧对应角点的像素坐标   
第一帧角点相机坐标系下的单位球坐标  
待求当前帧对应角点的单位球坐标
*/
// 此函数为选择的特征点计算光流  Compute optical flow (Lucas Kanade) for selected keypoints
void trackKlt(
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities)
{
  const double klt_win_size = 30.0;
  const int klt_max_iter = 30;
  const double klt_eps = 0.001;
  vector<uchar> status;
  vector<float> error;
  vector<float> min_eig_vec;
  
  /*迭代算法的终止条件  
  /一个是类型，第二个参数为迭代的最大次数，最后一个是特定的阈值
  /此类型表示迭 代终止条件为达到最大迭代次数终止或者迭代到阈值终止*/
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);

  // img_pyr_[0]就是对应的第零层金字塔  原图    当前帧第零层金字塔
  // 前一帧角点位置   待求当前帧角点位置(其初始化为前一帧的角点位置)
  // cv::OPTFLOW_USE_INITIAL_FLOW表示用px_cur已有的值作为初始估计，而不是使用px_ref作为初始估计
  cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                           px_ref, px_cur,
                           status, error,
                           cv::Size2i(klt_win_size, klt_win_size),
                           4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

  //上述方法调用完成后，得到了px_cur  当前帧的角点位置  status对于前一帧的角点，是否在当前帧中跟踪到了光流
  vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
  vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
  // 角点的相机坐标
  vector<Vector3d>::iterator f_ref_it = f_ref.begin();
  f_cur.clear(); f_cur.reserve(px_cur.size());
  disparities.clear(); disparities.reserve(px_cur.size());

  // 遍历前一帧的角点
  for(size_t i=0; px_ref_it != px_ref.end(); ++i)
  {
    // 如果在当前帧中没有跟踪到光流 把前一帧中的角点去掉
    if(!status[i])
    {
      px_ref_it = px_ref.erase(px_ref_it);
      px_cur_it = px_cur.erase(px_cur_it);
      f_ref_it = f_ref.erase(f_ref_it);
      continue;
    }

    // 当前帧的像素坐标转化为单位球坐标 c2f也调用了cam_的camtoworld方法 得到的是Z=1 然后归一化的三维坐标
    f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
    //两帧对应角点像素位置的距离
    disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());
    ++px_ref_it;
    ++px_cur_it;
    ++f_ref_it;
  }
}


// 计算单应矩阵，分解得到R,t,三角化得到在当前帧参考系下的三维坐标，通过重投影误差判断匹配角点的内点和离群点

// 参考帧单位球三维坐标  当前帧单位球三维坐标  焦距  重投影阈值  非待求内点  待求当前帧坐标系下的地图点坐标  待求的从参考帧到当前帧的位姿
void computeHomography(
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3& T_cur_from_ref)
{
  vector<Vector2d > uv_ref(f_ref.size());
  vector<Vector2d > uv_cur(f_cur.size());

  // 遍历对应角点，分别得到像素坐标
  for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
  {
    // project2d在math_utils.h中定义 应该是都除以z坐标  然后返回xy2D坐标
    // 所以对于单位球坐标f_ref和f_cur，调用了project2d函数之后，得到的是(X/Z,Y/Z)
    // 所以是归一化坐标??????
    uv_ref[i] = vk::project2d(f_ref[i]);
    uv_cur[i] = vk::project2d(f_cur[i]);
  }

  // 在homography.cpp中定义
  // 初始化  uv_ref  uv_cur为角点的归一化坐标
  vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
  //调用了cv中的findHomography方法 RANSAC+单应矩阵计算  得到单应矩阵  然后对单应矩阵进行了最优的Rt分解
  Homography.computeSE3fromMatches();
  vector<int> outliers;
  
  // math_utils.cpp中定义
  // 利用计算得到的单应矩阵，三角化得到匹配角点在当前帧参考系下的三维坐标xyz_in_cur,并通过重投影误差 找到所有匹配角点的内点和离群点
  // f_cur f_ref是单位球坐标
  vk::computeInliers(f_cur, f_ref,
                     Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
                     reprojection_threshold, focal_length,
                     xyz_in_cur, inliers, outliers);
  T_cur_from_ref = Homography.T_c2_from_c1;
}


} // namespace initialization
} // namespace svo
