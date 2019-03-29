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

#include <algorithm>
#include <vikit/math_utils.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <svo/global.h>
#include <svo/depth_filter.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/matcher.h>
#include <svo/config.h>
#include <svo/feature_detection.h>

namespace svo {

int Seed::batch_counter = 0;
int Seed::seed_counter = 0;

// 检测到的角点  平均深度   最小深度
Seed::Seed(Feature* ftr, float depth_mean, float depth_min) :
    batch_id(batch_counter),
    id(seed_counter++),
    ftr(ftr),
    a(10),
    b(10),
    // 均值  平均深度的倒数
    mu(1.0/depth_mean),
    // 深度范围为当前帧最近的深度的倒数
    z_range(1.0/depth_min),
    // 方差
    sigma2(z_range*z_range/36)
{}

DepthFilter::DepthFilter(feature_detection::DetectorPtr feature_detector, callback_t seed_converged_cb) :
    feature_detector_(feature_detector),
    seed_converged_cb_(seed_converged_cb),
    seeds_updating_halt_(false),
    thread_(NULL),
    new_keyframe_set_(false),
    new_keyframe_min_depth_(0.0),
    new_keyframe_mean_depth_(0.0)
{}

DepthFilter::~DepthFilter()
{
  stopThread();
  SVO_INFO_STREAM("DepthFilter destructed.");
}

// 调用updateSeedsLoop方法
void DepthFilter::startThread()
{
  thread_ = new boost::thread(&DepthFilter::updateSeedsLoop, this);
}

void DepthFilter::stopThread()
{
  SVO_INFO_STREAM("DepthFilter stop thread invoked.");
  if(thread_ != NULL)
  {
    SVO_INFO_STREAM("DepthFilter interrupt and join thread... ");
    seeds_updating_halt_ = true;
    thread_->interrupt();
    thread_->join();
    thread_ = NULL;
  }
}

// 利用当前帧更新种子点
void DepthFilter::addFrame(FramePtr frame)
{
  if(thread_ != NULL)
  {
    {
      lock_t lock(frame_queue_mut_);
      // 只保证frame_queue_有两帧?????????????????????
      if(frame_queue_.size() > 2)
        frame_queue_.pop();
      frame_queue_.push(frame);
    }
    
    seeds_updating_halt_ = false;
    frame_queue_cond_.notify_one();
  }
  else
  	// 更新种子点的深度  如果深度收敛 成为新的地图点
    updateSeeds(frame);
}



// 当前帧  当前帧坐标系下关联地图点的平均深度   最小深度的一半
void DepthFilter::addKeyframe(FramePtr frame, double depth_mean, double depth_min)
{
  new_keyframe_min_depth_ = depth_min;
  new_keyframe_mean_depth_ = depth_mean;
  if(thread_ != NULL)
  {
    new_keyframe_ = frame;
    new_keyframe_set_ = true;
    seeds_updating_halt_ = true;
    // ??????????????????这是在干啥
    // 线程之间的通信
    frame_queue_cond_.notify_one();
  }
  else
    initializeSeeds(frame);
}

void DepthFilter::initializeSeeds(FramePtr frame)
{
  Features new_features;
  // 在feature_detection.cpp中
  // 已经和上一帧完美匹配的特征点附件  禁止新建角点
  feature_detector_->setExistingFeatures(frame->fts_);

  // 进行当前帧的角点检测   阈值设置为20.0  得到新的角点 new_features
  feature_detector_->detect(frame.get(), frame->img_pyr_,
                            Config::triangMinCornerScore(), new_features);

  // initialize a seed for every new feature
  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_); // by locking the updateSeeds function stops
  ++Seed::batch_counter;
  // 对于每一个新检测到的角点  创建一个深度种子 
  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    seeds_.push_back(Seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));
  });

  if(options_.verbose)
    SVO_INFO_STREAM("DepthFilter: Initialized "<<new_features.size()<<" new seeds");
  seeds_updating_halt_ = false;
}



//移除关键帧
void DepthFilter::removeKeyframe(FramePtr frame)
{
  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_);
  std::list<Seed>::iterator it=seeds_.begin();
  size_t n_removed = 0;
  
  // 对于没有收敛的种子点
  while(it!=seeds_.end())
  {
  	// 如果种子点是由该帧产生的   将种子点删除
    if(it->ftr->frame == frame.get())
    {
      it = seeds_.erase(it);
      ++n_removed;
    }
    else
      ++it;
  }
  seeds_updating_halt_ = false;
}

void DepthFilter::reset()
{
  seeds_updating_halt_ = true;
  {
    lock_t lock(seeds_mut_);
    seeds_.clear();
  }
  lock_t lock();
  while(!frame_queue_.empty())
    frame_queue_.pop();
  seeds_updating_halt_ = false;

  if(options_.verbose)
    SVO_INFO_STREAM("DepthFilter: RESET.");
}


// 深度滤波器线程
void DepthFilter::updateSeedsLoop()
{
  while(!boost::this_thread::interruption_requested())
  {
    FramePtr frame;
    {
      lock_t lock(frame_queue_mut_);
      while(frame_queue_.empty() && new_keyframe_set_ == false)
        frame_queue_cond_.wait(lock);
      if(new_keyframe_set_)
      {
        new_keyframe_set_ = false;
        seeds_updating_halt_ = false;
        clearFrameQueue();
        frame = new_keyframe_;
      }
      else
      {
        frame = frame_queue_.front();
        frame_queue_.pop();
      }
    }
    updateSeeds(frame);
    if(frame->isKeyframe())
      initializeSeeds(frame);
  }
}




// 对于新进来的帧  利用极线搜索得到在当前帧中的最佳匹配  三角化得到地图点的深度  
// 利用地图点的深度和不确定性 更新地图点的均值 方差 a b 
// 如果地图点的深度收敛  新建地图点  删除种子点
void DepthFilter::updateSeeds(FramePtr frame)
{
  // update only a limited number of seeds, because we don't have time to do it
  // for all the seeds in every frame!
  size_t n_updates=0, n_failed_matches=0, n_seeds = seeds_.size();
  
  lock_t lock(seeds_mut_);
  std::list<Seed>::iterator it=seeds_.begin();


  const double focal_length = frame->cam_->errorMultiplier2();
  double px_noise = 1.0;
  // 这个是 如果当前帧得到的对应像素 误差在一个像素之内，对应的角度
  // ref:https://www.cnblogs.com/wxt11/p/7097250.html
  double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; // law of chord (sehnensatz)

  // seeds_是<Seed>类型的数组  Seed类型在depth_filter.h中定义
  while( it!=seeds_.end())
  {
    // set this value true when seeds updating should be interrupted
    if(seeds_updating_halt_)
      return;

    // check if seed is not already too old
    if((Seed::batch_counter - it->batch_id) > options_.max_n_kfs) {
      it = seeds_.erase(it);
      continue;
    }

    // check if point is visible in the current image
    // seed的frame是什么时候初始化的??????????????????????
    // 当前帧到参考帧的相对位姿
    SE3 T_ref_cur = it->ftr->frame->T_f_w_ * frame->T_f_w_.inverse();
    
    // 参考帧的坐标  *参考帧到当前帧的位姿   得到当前帧的坐标？？
    // 也就是将参考帧的地图点 投影到当前帧
    const Vector3d xyz_f(T_ref_cur.inverse()*(1.0/it->mu * it->ftr->f) );
    
    if(xyz_f.z() < 0.0)  {
      ++it; // behind the camera
      continue;
    }
    
    // seed在当前帧中是否可见
    if(!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>())) {
      ++it; // point does not project in image
      continue;
    }

    // we are using inverse depth coordinates
    // 利用深度的均值和方差得到最小逆深度和最大逆深度
    float z_inv_min = it->mu + sqrt(it->sigma2);
    float z_inv_max = max(it->mu - sqrt(it->sigma2), 0.00000001f);
    double z;
    // 处理seed   通过极线搜索  得到当前帧中匹配的像素点  三角化 得到深度z
    if(!matcher_.findEpipolarMatchDirect(
        *it->ftr->frame, *frame, *it->ftr, 1.0/it->mu, 1.0/z_inv_min, 1.0/z_inv_max, z))
    {
    	// b的值越大，深度的不确定性越大
      it->b++; // increase outlier probability when no match was found
      ++it;
      ++n_failed_matches;
      continue;
    }



    // compute tau
    // tau: 其实是如果当前帧得到的对应匹配点有一个像素的误差，那么造成的地图点深度的差值tau
    double tau = computeTau(T_ref_cur, it->ftr->f, z, px_error_angle);
    // 根据计算的深度的不确定性 得到最小深度和最大深度倒数的平均
    double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));


    // update the estimate
    // 利用得到的深度和不确定性  更新种子点的参数：均值 方差  a  b
    updateSeed(1./z, tau_inverse*tau_inverse, &*it);
    ++n_updates;

    // 如果当前帧是关键帧
    // ????????????????????????不是只有当前帧是非关键帧的时候  才会更新种子点么
    if(frame->isKeyframe())
    {
      // The feature detector should not initialize new seeds close to this location
      // 不允许在当前像素点的周围新建角点????????????????
      feature_detector_->setGridOccpuancy(matcher_.px_cur_);
    }



    // if the seed has converged, we initialize a new candidate point and remove the seed
    // seed_convergence_sigma2_thresh:200
    if(sqrt(it->sigma2) < it->z_range/options_.seed_convergence_sigma2_thresh)
    {
      assert(it->ftr->point == NULL); // TODO this should not happen anymore
      // it->mu是地图点以参考帧为坐标系收敛的深度值
      // 得到收敛的地图点的世界坐标
      Vector3d xyz_world(it->ftr->frame->T_f_w_.inverse() * (it->ftr->f * (1.0/it->mu)));
      // 得到新的地图点
      Point* point = new Point(xyz_world, it->ftr);
      it->ftr->point = point;
      /* FIXME it is not threadsafe to add a feature to the frame here.
      if(frame->isKeyframe())
      {
        Feature* ftr = new Feature(frame.get(), matcher_.px_cur_, matcher_.search_level_);
        ftr->point = point;
        point->addFrameRef(ftr);
        frame->addFeature(ftr);
        it->ftr->frame->addFeature(it->ftr);
      }
      else
      */
      {
      	// ??????????????????这个是什么
        seed_converged_cb_(point, it->sigma2); // put in candidate list
      }
      it = seeds_.erase(it);
    }
    else if(isnan(z_inv_min))
    {
      SVO_WARN_STREAM("z_min is NaN");
      it = seeds_.erase(it);
    }
    else
      ++it;
  }
}

void DepthFilter::clearFrameQueue()
{
  while(!frame_queue_.empty())
    frame_queue_.pop();
}

void DepthFilter::getSeedsCopy(const FramePtr& frame, std::list<Seed>& seeds)
{
  lock_t lock(seeds_mut_);
  for(std::list<Seed>::iterator it=seeds_.begin(); it!=seeds_.end(); ++it)
  {
    if (it->ftr->frame == frame.get())
      seeds.push_back(*it);
  }
}

// 根据三角化得到的深度值和深度的不确定性  更新种子点
// x:三角化得到的深度值的倒数   depthmax和depthmin倒数的平均  
// ref:https://blog.csdn.net/wubaobao1993/article/details/84136458
void DepthFilter::updateSeed(const float x, const float tau2, Seed* seed)
{
  float norm_scale = sqrt(seed->sigma2 + tau2);
  if(std::isnan(norm_scale))
    return;
  
  
  boost::math::normal_distribution<float> nd(seed->mu, norm_scale);
  float s2 = 1./(1./seed->sigma2 + 1./tau2);
  float m = s2*(seed->mu/seed->sigma2 + x/tau2);
  
  // ????????????这里C1 C2怎么计算的?????????????????
  float C1 = seed->a/(seed->a+seed->b) * boost::math::pdf(nd, x);
  float C2 = seed->b/(seed->a+seed->b) * 1./seed->z_range;
  float normalization_constant = C1 + C2;
  C1 /= normalization_constant;
  C2 /= normalization_constant;
  

  float f = C1*(seed->a+1.)/(seed->a+seed->b+1.) + C2*seed->a/(seed->a+seed->b+1.);
  float e = C1*(seed->a+1.)*(seed->a+2.)/((seed->a+seed->b+1.)*(seed->a+seed->b+2.))
          + C2*seed->a*(seed->a+1.0f)/((seed->a+seed->b+1.0f)*(seed->a+seed->b+2.0f));

  // update parameters
  // 种子点新的均值
  float mu_new = C1*m+C2*seed->mu;
  // mu_new: 代码中是tao??????????????????????
  seed->sigma2 = C1*(s2 + m*m) + C2*(seed->sigma2 + seed->mu*seed->mu) - mu_new*mu_new;
  seed->mu = mu_new;
  seed->a = (e-f)/(f-e/f);
  seed->b = seed->a*(1.0f-f)/f;
}


// computeTau(T_ref_cur, it->ftr->f, z, px_error_angle)
// ref:https://www.cnblogs.com/wxt11/p/7097250.html
// px_error_angle: 如果当前帧得到的对应像素 误差在一个像素之内，对应的角度
// 当前帧到参考帧的相对位姿   参考帧角点的单位球坐标   角点对应地图点的深度   
double DepthFilter::computeTau(
      const SE3& T_ref_cur,
      const Vector3d& f,
      const double z,
      const double px_error_angle)
{
	// t是两个相机之间的距离
  Vector3d t(T_ref_cur.translation());
  // 参考帧坐标系下的三维坐标 -t得到地图点在当前帧中的位置
  Vector3d a = f*z-t;
  
  double t_norm = t.norm();
  double a_norm = a.norm();
  
  double alpha = acos(f.dot(t)/t_norm); // dot product
  double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
  double beta_plus = beta + px_error_angle;
  double gamma_plus = PI-alpha-beta_plus; // triangle angles sum to PI
  double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
  return (z_plus - z); // tau
}

} // namespace svo
