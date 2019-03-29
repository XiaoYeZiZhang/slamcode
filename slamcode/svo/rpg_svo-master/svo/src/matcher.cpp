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

#include <cstdlib>
#include <vikit/abstract_camera.h>
#include <vikit/vision.h>
#include <vikit/math_utils.h>
#include <vikit/patch_score.h>
#include <svo/matcher.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/config.h>
#include <svo/feature_alignment.h>

namespace svo {

namespace warp {

// 参考帧相机  当前帧相机  参考帧像素坐标   参考帧单位球坐标  参考帧相机到地图点的距离(深度)  参考帧到当前帧相对位姿  level(应该是0??? feature初始化的时候为0)
// 得到参考帧到当前帧的仿射变换  这个仿射变换应该是 参考帧中水平和垂直一个像素，在当前帧中移动的方向
void getWarpMatrixAffine(
    const vk::AbstractCamera& cam_ref,
    const vk::AbstractCamera& cam_cur,
    const Vector2d& px_ref,
    const Vector3d& f_ref,
    const double depth_ref,
    const SE3& T_cur_ref,
    const int level_ref,
    Matrix2d& A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const int halfpatch_size = 5;
  
  // 在参考帧坐标系下的地图点坐标
  const Vector3d xyz_ref(f_ref*depth_ref);
  
  // 特征点 在对应的层数上 取右边第五个像素 和下边第五个像素位置，映射到第零层
  // cam2world 将像素坐标转换为单位球三维坐标
  
  // 这里得到的xyz_du_ref和xyz_dv_ref应该可以理解成  参考帧角点水平和垂直方向差5个像素的两个像素点像素点的单位球坐标
  Vector3d xyz_du_ref(cam_ref.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)));
  Vector3d xyz_dv_ref(cam_ref.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)));
  
  // 将单位球坐标映射到三维空间  和空间点有相同的深度xyz_ref[2]
  xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2];
  
  
  // 到参考帧坐标系下 再根据相对位姿 映射到当前帧像素坐标
  
  // 利用参考帧地图点和相对位姿 得到投影到当前帧的像素坐标
  const Vector2d px_cur(cam_cur.world2cam(T_cur_ref*(xyz_ref)));
  
  const Vector2d px_du(cam_cur.world2cam(T_cur_ref*(xyz_du_ref)));
  const Vector2d px_dv(cam_cur.world2cam(T_cur_ref*(xyz_dv_ref)));
  
  // 5
  A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}


// 参考帧到当前帧的仿射变换
int getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  
  while(D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}


// 仿射矩阵  尺度下的参考帧   像素坐标   参考帧的尺度   最优尺度  3  patch[10*10] 
// patch最后得到的应该是当前帧对应角点的10*10个图块， 在参考帧中对应的位置
void warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int search_level,
    const int halfpatch_size,
    uint8_t* patch)
{
	// 10
  const int patch_size = halfpatch_size*2 ;
  // 从当前帧到参考帧的仿射变换
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if(isnan(A_ref_cur(0,0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }

  // Perform the warp on a larger patch.
  // int[10*10]
  uint8_t* patch_ptr = patch;
  
  // 在level上的参考帧像素坐标
  const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref);
  
  // 10
  for (int y=0; y<patch_size; ++y)
  {
    for (int x=0; x<patch_size; ++x, ++patch_ptr)
    {
    	
      Vector2f px_patch(x-halfpatch_size, y-halfpatch_size);
      // 恢复到level0
      px_patch *= (1<<search_level);
      // 对应在参考帧中的像素
      const Vector2f px(A_ref_cur*px_patch + px_ref_pyr);
      
      if (px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
        *patch_ptr = 0;
      else
        *patch_ptr = (uint8_t) vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

} // namespace warp


/*
depthFromTriangulation(T_cur_ref, ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth)
参考帧到当前帧的相对位姿   参考帧角点的单位球坐标    当前帧匹配角点的单位球坐标  待求深度
*/
// ref:https://blog.csdn.net/luoshi006/article/details/80792043
bool depthFromTriangulation(
    const SE3& T_search_ref,
    const Vector3d& f_ref,
    const Vector3d& f_cur,
    double& depth)
{
  Matrix<double,3,2> A; 
  A << T_search_ref.rotation_matrix() * f_ref, f_cur;
  const Matrix2d AtA = A.transpose()*A;
  if(AtA.determinant() < 0.000001)
    return false;
  const Vector2d depth2 = - AtA.inverse()*A.transpose()*T_search_ref.translation();
  // 返回的是交点在参考帧中的深度
  depth = fabs(depth2[0]);
  return true;
}




// patch_size_ = 8
// 从10*10的图块中选取8*8的图块
void Matcher::createPatchFromPatchWithBorder()
{
	// patch_[8*8] 在matcher.h中
  uint8_t* ref_patch_ptr = patch_;
  // y = 1 y < 9 y++,ref+=8
  for(int y=1; y<patch_size_+1; ++y, ref_patch_ptr += patch_size_)
  {
    uint8_t* ref_patch_border_ptr = patch_with_border_ + y*(patch_size_+2) + 1;
    for(int x=0; x<patch_size_; ++x)
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
  }
}


// 三维点  当前帧  地图点在当前帧的像素坐标
bool Matcher::findMatchDirect(
    const Point& pt,
    const Frame& cur_frame,
    Vector2d& px_cur)
{
	// point.cpp中   当前帧的位置
	// 根据观测方向  得到与当前帧最相近的关键帧对应该地图点的特征ref_ftr_
  if(!pt.getCloseViewObs(cur_frame.pos(), ref_ftr_))
    return false;

  // halfpatch_size = 4  这里参数是6
  // 如果该特征点在该关键帧中能取到6*6大小的像素块
  if(!ref_ftr_->frame->cam_->isInFrame(
      ref_ftr_->px.cast<int>()/(1<<ref_ftr_->level), halfpatch_size_+2, ref_ftr_->level))
    return false;

  // warp affine
  // 得到参考帧到当前帧的仿射变换
  // 仿射矩阵A 就是把参考帧上的图块在自己对应的层数上，转换到当前帧的第零层上
  // 参考帧水平和垂直方向移动一个像素，在当前帧中的移动情况
  warp::getWarpMatrixAffine(
      *ref_ftr_->frame->cam_, *cur_frame.cam_, ref_ftr_->px, ref_ftr_->f,
      (ref_ftr_->frame->pos() - pt.pos_).norm(),
      cur_frame.T_f_w_ * ref_ftr_->frame->T_f_w_.inverse(), ref_ftr_->level, A_cur_ref_);
  
  // 所以仿射矩阵A其实就是面积放大的比例
  // 根据仿射变换的行列式 得到最优搜索尺度 
  search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);
  
  // 根据仿射变换  从参考帧上的特征点附近取一些像素 ，映射到当前帧正好组成10*10的图块
  // ???????????????????????????
  // patch_with_border_在matcher.h中定义  为int[10*10]
  // 就是地图点对应在当前帧中的像素，在10*10的图块中  对应参考帧中的哪些像素
  warp::warpAffine(A_cur_ref_, ref_ftr_->frame->img_pyr_[ref_ftr_->level], ref_ftr_->px,
                   ref_ftr_->level, search_level_, halfpatch_size_+1, patch_with_border_);
  // 取8*8
  createPatchFromPatchWithBorder();



  // patch_和patch_with_border是个啥东西
  // px_cur should be set
  // 地图点在当前帧最优尺度下的像素坐标
  Vector2d px_scaled(px_cur/(1<<search_level_));
  bool success = false;
  
  if(ref_ftr_->type == Feature::EDGELET)
  {
    Vector2d dir_cur(A_cur_ref_*ref_ftr_->grad);
    dir_cur.normalize();
    success = feature_alignment::align1D(
          cur_frame.img_pyr_[search_level_], dir_cur.cast<float>(),
          patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
  }
  else
  {
  	// px_scaled是在对应尺度下  当前帧对应的像素坐标
    success = feature_alignment::align2D(
      cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
      options_.align_max_iter, px_scaled);
  }
  // 更新后的当前帧像素坐标
  px_cur = px_scaled * (1<<search_level_);
  return success;
}



// 处理seed
/**it->ftr->frame, *frame, *it->ftr, 1.0/it->mu, 1.0/z_inv_min, 1.0/z_inv_max, z*/
// seed所属的参考帧  用于更新seed的当前帧  角点  深度的平均值  深度最小值   深度最大值  待求深度
bool Matcher::findEpipolarMatchDirect(
    const Frame& ref_frame,
    const Frame& cur_frame,
    const Feature& ref_ftr,
    const double d_estimate,
    const double d_min,
    const double d_max,
    double& depth)
{
	// 从参考帧到当前帧的相对位姿
  SE3 T_cur_ref = cur_frame.T_f_w_ * ref_frame.T_f_w_.inverse();
  int zmssd_best = PatchScore::threshold();
  Vector2d uv_best;

  // Compute start and end of epipolar line in old_kf for match search, on unit plane!
  // ref_ftr.f*d_min 和ref_ftr.f*d_max应该是这样的： 在参考帧的一个特征点和相机光心连线上  三维地图点可能的范围
  // 将这段范围的两个端点投影到当前帧图像上  连线就是极线的方向
  // projected2d得到的是 X/Z,Y/Z
  Vector2d A = vk::project2d(T_cur_ref * (ref_ftr.f*d_min));
  Vector2d B = vk::project2d(T_cur_ref * (ref_ftr.f*d_max));
  epi_dir_ = A - B;

  // Compute affine warp matrix
  warp::getWarpMatrixAffine(
      *ref_frame.cam_, *cur_frame.cam_, ref_ftr.px, ref_ftr.f,
      d_estimate, T_cur_ref, ref_ftr.level, A_cur_ref_);

  // feature pre-selection
  reject_ = false;
  // matcher.h中epi_search_edgelet_filtering设置为true
  // Feature::EDGELET表示边缘特征 ????????怎么判断是边缘的???梯度么
  if(ref_ftr.type == Feature::EDGELET && options_.epi_search_edgelet_filtering)
  {
  	// 利用仿射矩阵 将参考帧的梯度映射到当前帧
    const Vector2d grad_cur = (A_cur_ref_ * ref_ftr.grad).normalized();
    // 计算当前帧梯度和极线的夹角
    const double cosangle = fabs(grad_cur.dot(epi_dir_.normalized()));
    // 如果夹角cos大于0.7  不进行搜索..??????????原因???????????
    // 认为沿着极线找，图块像素也不会变化很大
    if(cosangle < options_.epi_search_edgelet_max_angle) {
      reject_ = true;
      return false;
    }
  }

  search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);

  // Find length of search range on epipolar line
  // 将地图点深度两个极值点 投影到当前帧，转换为像素坐标
  Vector2d px_A(cur_frame.cam_->world2cam(A));
  Vector2d px_B(cur_frame.cam_->world2cam(B));
  // 得到极线的长度
  epi_length_ = (px_A-px_B).norm() / (1<<search_level_);


  // 利用这个扭变的仿射矩阵，将参考帧角点周围的10*10图像块，在当前帧中找到对应的10*10图像块(因为会有扭变，所以要找到对应的图像块(就不一定是矩形了))
  // Warp reference patch at ref_level
  warp::warpAffine(A_cur_ref_, ref_frame.img_pyr_[ref_ftr.level], ref_ftr.px,
                   ref_ftr.level, search_level_, halfpatch_size_+1, patch_with_border_);
  createPatchFromPatchWithBorder();


  // 如果极线长度小于两个像素
  if(epi_length_ < 2.0)
  {
  	// 取中间像素  得到在当前帧中搜索到的与参考帧对应的像素点
    px_cur_ = (px_A+px_B)/2.0;
    // 转换到相应尺度下
    Vector2d px_scaled(px_cur_/(1<<search_level_));
    
    bool res;
    // false
    if(options_.align_1d)
      res = feature_alignment::align1D(
          cur_frame.img_pyr_[search_level_], (px_A-px_B).cast<float>().normalized(),
          patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
    else
    	// align2D 函数通过重投影误差  进一步优化在当前帧的投影像素位置
    	// 优化后的像素位置存放在变量px_scaled中
      res = feature_alignment::align2D(
          cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
          options_.align_max_iter, px_scaled);

    if(res)
    {
    	// 得到当前帧图像上匹配上的像素坐标
      px_cur_ = px_scaled*(1<<search_level_);
      // 利用三角化 得到地图点的深度值 depth
      if(depthFromTriangulation(T_cur_ref, ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth))
        return true;
    }
    return false;
  }


  // 一共要探索的步长
  size_t n_steps = epi_length_/0.7; // one step per pixel
  // 每一步要走的像素数目
  Vector2d step = epi_dir_/n_steps;

  // 1000
  if(n_steps > options_.max_epi_search_steps)
  {
    printf("WARNING: skip epipolar search: %zu evaluations, px_lenght=%f, d_min=%f, d_max=%f.\n",
           n_steps, epi_length_, d_min, d_max);
    return false;
  }


  // for matching, precompute sum and sum2 of warped reference patch
  int pixel_sum = 0;
  int pixel_sum_square = 0;
  PatchScore patch_score(patch_);

  // now we sample along the epipolar line
  // 在极线上进行采样
  
  // 采样得到的像素位置
  Vector2d uv = B-step;
  Vector2i last_checked_pxi(0,0);
  ++n_steps;
  
  for(size_t i=0; i<n_steps; ++i, uv+=step)
  {
  	// 转换到单位球坐标
    Vector2d px(cur_frame.cam_->world2cam(uv));
    // 放入相应的level中
    Vector2i pxi(px[0]/(1<<search_level_)+0.5,
                 px[1]/(1<<search_level_)+0.5); // +0.5 to round to closest int

    if(pxi == last_checked_pxi)
      continue;
    last_checked_pxi = pxi;


    // check if the patch is full within the new frame
    if(!cur_frame.cam_->isInFrame(pxi, patch_size_, search_level_))
      continue;

    // TODO interpolation would probably be a good idea
    // 在极线上的像素周围得到像素块
    uint8_t* cur_patch_ptr = cur_frame.img_pyr_[search_level_].data
                             + (pxi[1]-halfpatch_size_)*cur_frame.img_pyr_[search_level_].cols
                             + (pxi[0]-halfpatch_size_);
    // 这里计算的是 当前帧像素对应的像素块  和参考帧像素对应的像素块 的相似度?????????????????????
    int zmssd = patch_score.computeScore(cur_patch_ptr, cur_frame.img_pyr_[search_level_].cols);

    if(zmssd < zmssd_best) {
      zmssd_best = zmssd;
      uv_best = uv;
    }
  }
 
  // threshold(): 2000*patch_area_
  if(zmssd_best < PatchScore::threshold())
  {
  	// true
    if(options_.subpix_refinement)
    {
      px_cur_ = cur_frame.cam_->world2cam(uv_best);
      Vector2d px_scaled(px_cur_/(1<<search_level_));
      bool res;
      if(options_.align_1d)
        res = feature_alignment::align1D(
            cur_frame.img_pyr_[search_level_], (px_A-px_B).cast<float>().normalized(),
            patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
      else
      	// 通过重投影误差  优化当前帧对应像素的位置
        res = feature_alignment::align2D(
            cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
            options_.align_max_iter, px_scaled);
      if(res)
      {
      	// 进行三角化  得到地图点的深度
        px_cur_ = px_scaled*(1<<search_level_);
        if(depthFromTriangulation(T_cur_ref, ref_ftr.f, cur_frame.cam_->cam2world(px_cur_), depth))
          return true;
      }
      return false;
    }
    
    // 得到当前帧对应像素的单位球坐标
    px_cur_ = cur_frame.cam_->world2cam(uv_best);
    
    // 利用uv_best转换为单位球坐标
    if(depthFromTriangulation(T_cur_ref, ref_ftr.f, vk::unproject2d(uv_best).normalized(), depth))
      return true;
  }
  return false;
}

} // namespace svo
