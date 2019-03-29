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
#include <svo/sparse_img_align.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/config.h>
#include <svo/point.h>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <vikit/math_utils.h>

namespace svo {

/* 4 2 30 gaussNewton  false  false*/
SparseImgAlign::SparseImgAlign(
    int max_level, int min_level, int n_iter,
    Method method, bool display, bool verbose) :
        display_(display),
        max_level_(max_level),
        min_level_(min_level)
{
  n_iter_ = n_iter;
  n_iter_init_ = n_iter_;
  method_ = method;
  verbose_ = verbose;
  eps_ = 0.000001;
}



// 上一帧   当前帧
// 重定位中:  共视关键帧   当前帧
size_t SparseImgAlign::run(FramePtr ref_frame, FramePtr cur_frame)
{
  reset();

  if(ref_frame->fts_.empty())
  {
    SVO_WARN_STREAM("SparseImgAlign: no features to track!");
    return 0;
  }

  ref_frame_ = ref_frame;
  cur_frame_ = cur_frame;
  
  // patch_area_是16
  // Mat是 角点数量的行 16列
  ref_patch_cache_ = cv::Mat(ref_frame_->fts_.size(), patch_area_, CV_32F);
  // 6 * (角点数量*16)大小
  jacobian_cache_.resize(Eigen::NoChange, ref_patch_cache_.rows*patch_area_); //(角点数量)
  visible_fts_.resize(ref_patch_cache_.rows, false); // TODO: should it be reset at each level?

  // 参考帧到当前帧的位姿
  SE3 T_cur_from_ref(cur_frame_->T_f_w_ * ref_frame_->T_f_w_.inverse());

  // 4 2  金字塔层数  从小的图像开始，逐步求精
  for(level_=max_level_; level_>=min_level_; --level_)
  {
    mu_ = 0.1;
    jacobian_cache_.setZero();
    have_ref_patch_cache_ = false;
    if(verbose_)
      printf("\nPYRAMID LEVEL %i\n---------------\n", level_);
      
    // 是在nlls_solver_impl.hpp中实现
    // 优化相对位姿T_cur_from_ref  最终得到优化后的位姿
    // 在不同scale下进行优化,每次优化的值都存放在T_cur_from_ref中，等于是在尺度增大的情况下不断调优
    optimize(T_cur_from_ref);
  }
  // 通过优化后的相对位姿更新当前帧的位姿
  cur_frame_->T_f_w_ = T_cur_from_ref * ref_frame_->T_f_w_;

  return n_meas_/patch_area_;
}

Matrix<double, 6, 6> SparseImgAlign::getFisherInformation()
{
  double sigma_i_sq = 5e-4*255*255; // image noise
  Matrix<double,6,6> I = H_/sigma_i_sq;
  return I;
}

// ref:http://www.cnblogs.com/ilekoaiq/p/8659631.html     slambook 205页
// 预先计算雅克比矩阵
void SparseImgAlign::precomputeReferencePatches()
{
  // 2+1 = 3
  // border = 3
  const int border = patch_halfsize_+1;
  // 得到相应尺度下的参考帧
  const cv::Mat& ref_img = ref_frame_->img_pyr_.at(level_);
  	
  const int stride = ref_img.cols;
  // scale是为了转换到对应的金字塔图像上
  const float scale = 1.0f/(1<<level_);
  
  //  参考帧在第一帧中的位移
  const Vector3d ref_pos = ref_frame_->pos();
  // 焦距
  const double focal_length = ref_frame_->cam_->errorMultiplier2();
  
  size_t feature_counter = 0;
  
  // visible_fts_：(角点数量)大小
  std::vector<bool>::iterator visiblity_it = visible_fts_.begin();
  // 遍历参考帧的角点
  for(auto it=ref_frame_->fts_.begin(), ite=ref_frame_->fts_.end();
      it!=ite; ++it, ++feature_counter, ++visiblity_it)
  {
    // check if reference with patch size is within image
    const float u_ref = (*it)->px[0]*scale;
    const float v_ref = (*it)->px[1]*scale;
    //floorf返回小于等于它的最大整数
    // 得到参考帧中的特征点在对应金字塔层数下的像素坐标(要进行取整得到像素点)
    const int u_ref_i = floorf(u_ref);
    const int v_ref_i = floorf(v_ref);
    
    
    if((*it)->point == NULL || u_ref_i-border < 0 || v_ref_i-border < 0 || u_ref_i+border >= ref_img.cols || v_ref_i+border >= ref_img.rows)
      continue;
    *visiblity_it = true;

    // ???????????怎么得到的在参考帧坐标系下的地图点
    // 要优化的残差是，参考帧上的特征点的图块与投影到当前帧上的位置上的图块的亮度残差。
    // 投影位置是，参考帧中的特征点延伸到三维空间中到与对应的地图点深度一样的位置，然后投影到当前帧。
    // cannot just take the 3d points coordinate because of the reprojection errors in the reference image!!!
    const double depth(((*it)->point->pos_ - ref_pos).norm());
    // 参考帧坐标系下的地图点
    const Vector3d xyz_ref((*it)->f*depth);

    // evaluate projection jacobian
    Matrix<double,2,6> frame_jac;
    // frame.h
    // 根据参考帧坐标系下的地图点坐标  构建雅克比矩阵frame_jac
    // 雅克比矩阵描述的是重投影误差关于相机位姿李代数的一阶变化关系
    Frame::jacobian_xyz2uv(xyz_ref, frame_jac);

    // compute bilateral interpolation weights for reference image
    // 因为取了整数  所以要计算一下周围像素的权重????????
    const float subpix_u_ref = u_ref-u_ref_i;
    const float subpix_v_ref = v_ref-v_ref_i;
    
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    size_t pixel_counter = 0;
    // 存储每个角点周围4*4区域的光度值
    float* cache_ptr = reinterpret_cast<float*>(ref_patch_cache_.data) + patch_area_*feature_counter;
    
    // patch_size_ = 4  patch_halfsize_ = 2
    for(int y=0; y<patch_size_; ++y)
    {
      // stride 是相应尺度下image的列数
      // u_ref_i  v_ref_i是角点在金字塔相应层数下的坐标
      // ref_img_ptr应该是在相应层数下 角点4*4patch最左边的指针  通过x那里 ref_img_ptr++来向右进4个
      uint8_t* ref_img_ptr = (uint8_t*) ref_img.data + (v_ref_i+y-patch_halfsize_)*stride + (u_ref_i-patch_halfsize_);
      // 逐行
      for(int x=0; x<patch_size_; ++x, ++ref_img_ptr, ++cache_ptr, ++pixel_counter)
      {
        // precompute interpolated reference patch color
        *cache_ptr = w_ref_tl*ref_img_ptr[0] + w_ref_tr*ref_img_ptr[1] + w_ref_bl*ref_img_ptr[stride] + w_ref_br*ref_img_ptr[stride+1];

        // we use the inverse compositional: thereby we can take the gradient always at the same position
        // get gradient of warped image (~gradient at warped position)
        // w_ref_tl   w_ref_tr  w_ref_bl w_ref_br 是左上右上 左下右下的权重
        // ?????????????这里本来就是在小的尺度上进行的，为什么还要进行插值???????????????
        // 因为投影到该层金字塔的时候 像素不是整数，需要将周围四个像素点进行一下加权??得到梯度值
        float dx = 0.5f * ((w_ref_tl*ref_img_ptr[1] + w_ref_tr*ref_img_ptr[2] + w_ref_bl*ref_img_ptr[stride+1] + w_ref_br*ref_img_ptr[stride+2])
                          -(w_ref_tl*ref_img_ptr[-1] + w_ref_tr*ref_img_ptr[0] + w_ref_bl*ref_img_ptr[stride-1] + w_ref_br*ref_img_ptr[stride]));
                          
                          
        float dy = 0.5f * ((w_ref_tl*ref_img_ptr[stride] + w_ref_tr*ref_img_ptr[1+stride] + w_ref_bl*ref_img_ptr[stride*2] + w_ref_br*ref_img_ptr[stride*2+1])
                          -(w_ref_tl*ref_img_ptr[-stride] + w_ref_tr*ref_img_ptr[1-stride] + w_ref_bl*ref_img_ptr[0] + w_ref_br*ref_img_ptr[1]));

        // cache the jacobian
        // pixel_counter是一个角点中的16个
        // feature_counter是每个角点
        jacobian_cache_.col(feature_counter*patch_area_ + pixel_counter) =
            (dx*frame_jac.row(0) + dy*frame_jac.row(1))*(focal_length / (1<<level_));
      }
    }
  }
  have_ref_patch_cache_ = true;
}





// true false 
// 函数返回的是平均重投影误差
double SparseImgAlign::computeResiduals(
    const SE3& T_cur_from_ref,
    bool linearize_system,
    bool compute_weight_scale)
{
  // Warp the (cur)rent image such that it aligns with the (ref)erence image
  
  // 得到在当前尺度下的当前帧图像
  const cv::Mat& cur_img = cur_frame_->img_pyr_.at(level_);

  if(linearize_system && display_)
    resimg_ = cv::Mat(cur_img.size(), CV_32F, cv::Scalar(0));

  // 如果没有预存雅克比矩阵 先进行计算
  if(have_ref_patch_cache_ == false)
  	// 预先计算雅克比矩阵(参考帧)
    precomputeReferencePatches();

  // compute the weights on the first iteration
  std::vector<float> errors;
  // false
  if(compute_weight_scale)
    errors.reserve(visible_fts_.size());
    
  const int stride = cur_img.cols;
  const int border = patch_halfsize_+1;
  const float scale = 1.0f/(1<<level_);
  const Vector3d ref_pos(ref_frame_->pos());
  float chi2 = 0.0;
  size_t feature_counter = 0; // is used to compute the index of the cached jacobian
  std::vector<bool>::iterator visiblity_it = visible_fts_.begin();
  
  // 遍历参考帧的角点
  for(auto it=ref_frame_->fts_.begin(); it!=ref_frame_->fts_.end();
      ++it, ++feature_counter, ++visiblity_it)
  {
    // check if feature is within image
    if(!*visiblity_it)
      continue;

    // compute pixel location in cur img
    // 参考帧坐标系下的地图点(参考帧像素点不变 所以地图点也不变  雅克比可以预存)
    const double depth = ((*it)->point->pos_ - ref_pos).norm();
    const Vector3d xyz_ref((*it)->f*depth);
    
    // 利用相对位姿得到的在当前帧坐标系下的地图点
    // 所以在整个迭代的过程中  发生变化的是T_cur_from_ref！！！！！！！！！！！！！！！
    // 也就是函数的参数，在nlls_volver_impl.hpp中是model
    const Vector3d xyz_cur(T_cur_from_ref * xyz_ref);
    
    // 在相应尺度下 投影到当前帧的像素坐标
    const Vector2f uv_cur_pyr(cur_frame_->cam_->world2cam(xyz_cur).cast<float>() * scale);
    const float u_cur = uv_cur_pyr[0];
    const float v_cur = uv_cur_pyr[1];
    
    const int u_cur_i = floorf(u_cur);
    const int v_cur_i = floorf(v_cur);

    // check if projection is within the image
    if(u_cur_i < 0 || v_cur_i < 0 || u_cur_i-border < 0 || v_cur_i-border < 0 || u_cur_i+border >= cur_img.cols || v_cur_i+border >= cur_img.rows)
      continue;

    // compute bilateral interpolation weights for the current image
    const float subpix_u_cur = u_cur-u_cur_i;
    const float subpix_v_cur = v_cur-v_cur_i;
    //???????????????这些权重是怎么来的????????????????
    const float w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
    const float w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
    const float w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
    const float w_cur_br = subpix_u_cur * subpix_v_cur;
    
    // 得到在参考帧中该角点对应的亮度值
    float* ref_patch_cache_ptr = reinterpret_cast<float*>(ref_patch_cache_.data) + patch_area_*feature_counter;
    
    
    size_t pixel_counter = 0; // is used to compute the index of the cached jacobian
    // 遍历当前帧的 4*4图像块
    for(int y=0; y<patch_size_; ++y)
    {
      uint8_t* cur_img_ptr = (uint8_t*) cur_img.data + (v_cur_i+y-patch_halfsize_)*stride + (u_cur_i-patch_halfsize_);

      for(int x=0; x<patch_size_; ++x, ++pixel_counter, ++cur_img_ptr, ++ref_patch_cache_ptr)
      {
        // compute residual
        // 计算光度误差  还是用的插值??????????????
        const float intensity_cur = w_cur_tl*cur_img_ptr[0] + w_cur_tr*cur_img_ptr[1] + w_cur_bl*cur_img_ptr[stride] + w_cur_br*cur_img_ptr[stride+1];
        const float res = intensity_cur - (*ref_patch_cache_ptr);

        // used to compute scale for robust cost
        if(compute_weight_scale)
          errors.push_back(fabsf(res));

        // robustification
        float weight = 1.0;
        if(use_weights_) {
          weight = weight_function_->value(res/scale_);
        }

        chi2 += res*res*weight;
        // n_means是计算了偏差的像素点数目
        n_meas_++;

        // true
        if(linearize_system)
        {
          // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
          const Vector6d J(jacobian_cache_.col(feature_counter*patch_area_ + pixel_counter));
          // noalias()不发生混淆
          H_.noalias() += J*J.transpose()*weight;
          Jres_.noalias() -= J*res*weight;
          if(display_)
            resimg_.at<float>((int) v_cur+y-patch_halfsize_, (int) u_cur+x-patch_halfsize_) = res/255.0;
        }
      }
    }
  }

  // compute the weights on the first iteration
  if(compute_weight_scale && iter_ == 0)
    scale_ = scale_estimator_->compute(errors);

  return chi2/n_meas_;
}

int SparseImgAlign::solve()
{
	// Ax = b
	// 用A.ldlt().solve(b)可以求解x  其中ldlt()是LDLT分解 A = LDLT
  x_ = H_.ldlt().solve(Jres_);
  if((bool) std::isnan((double) x_[0]))
    return 0;
  return 1;
}


// ?????????????????????????????????位姿的更新是*-x??????????????
void SparseImgAlign::update(
    const ModelType& T_curold_from_ref,
    ModelType& T_curnew_from_ref)
{
  T_curnew_from_ref =  T_curold_from_ref * SE3::exp(-x_);
}

void SparseImgAlign::startIteration()
{}

void SparseImgAlign::finishIteration()
{
  if(display_)
  {
    cv::namedWindow("residuals", CV_WINDOW_AUTOSIZE);
    cv::imshow("residuals", resimg_*10);
    cv::waitKey(0);
  }
}

} // namespace svo

