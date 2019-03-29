/*
 * homography.cpp
 * Adaptation of PTAM-GPL HomographyInit class.
 * https://github.com/Oxford-PTAM/PTAM-GPL
 * Licence: GPLv3
 * Copyright 2008 Isis Innovation Limited
 *
 *  Created on: Sep 2, 2012
 *      by: cforster
 */

#include <vikit/homography.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>

namespace vk {


//vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
// 参考帧角点的归一化坐标   当前帧角点的归一化坐标
Homography::
Homography(const vector<Vector2d, aligned_allocator<Vector2d> >& _fts1,
           const vector<Vector2d, aligned_allocator<Vector2d> >& _fts2,
           double _error_multiplier2,
           double _thresh_in_px) :
   thresh(_thresh_in_px),
   error_multiplier2(_error_multiplier2),
   fts_c1(_fts1),
   fts_c2(_fts2)
{
}

void Homography::
calcFromPlaneParams(const Vector3d& n_c1, const Vector3d& xyz_c1)
{
  double d = n_c1.dot(xyz_c1); // normal distance from plane to KF
  H_c2_from_c1 = T_c2_from_c1.rotation_matrix() + (T_c2_from_c1.translation()*n_c1.transpose())/d;
}

// 计算单应矩阵
void Homography::
calcFromMatches()
{
  //参考帧  当前帧
  vector<cv::Point2f> src_pts(fts_c1.size()), dst_pts(fts_c1.size());
  // 遍历对应角点
  for(size_t i=0; i<fts_c1.size(); ++i)
  {
    src_pts[i] = cv::Point2f(fts_c1[i][0], fts_c1[i][1]);
    dst_pts[i] = cv::Point2f(fts_c2[i][0], fts_c2[i][1]);
  }

  // TODO: replace this function to remove dependency from opencv!
  // ref:  https://blog.csdn.net/qq_25352981/article/details/51530751
  // 利用归一化坐标  RANSAC得到单应矩阵
  cv::Mat cvH = cv::findHomography(src_pts, dst_pts, CV_RANSAC, 2./error_multiplier2);
  	
  H_c2_from_c1(0,0) = cvH.at<double>(0,0);
  H_c2_from_c1(0,1) = cvH.at<double>(0,1);
  H_c2_from_c1(0,2) = cvH.at<double>(0,2);
  H_c2_from_c1(1,0) = cvH.at<double>(1,0);
  H_c2_from_c1(1,1) = cvH.at<double>(1,1);
  H_c2_from_c1(1,2) = cvH.at<double>(1,2);
  H_c2_from_c1(2,0) = cvH.at<double>(2,0);
  H_c2_from_c1(2,1) = cvH.at<double>(2,1);
  H_c2_from_c1(2,2) = cvH.at<double>(2,2);
}

size_t Homography::
computeMatchesInliers()
{
	
  inliers.clear(); inliers.resize(fts_c1.size());
  size_t n_inliers = 0;
  // 遍历所有匹配点的归一化坐标(X/Z,Y/Z)！！不是单位球坐标
  for(size_t i=0; i<fts_c1.size(); i++)
  {
    // unprojected: (X/Z,Y/Z,1)
    //project2d: X,Y,Z->X/Z,Y/Z  转换为当前帧坐标系下的归一化坐标
    // 将参考帧根据H矩阵算出在当前帧的投影，得到误差
    // ????????????error_multiplier2是焦距
    Vector2d projected = project2d(H_c2_from_c1 * unproject2d(fts_c1[i]));
    // 两个归一化坐标相减 得到重投影误差
    Vector2d e = fts_c2[i] - projected;
    double e_px = error_multiplier2 * e.norm();
    inliers[i] = (e_px < thresh);
    n_inliers += inliers[i];
  }
  return n_inliers;

}

// 利用对应角点恢复单应矩阵
bool Homography::
computeSE3fromMatches()
{
  //利用cv计算单应矩阵 保存在H_c2_from_c1中
  calcFromMatches();
  //利用单应矩阵恢复R 和 t:  decomp.R 和 decomp.t  decomp.T
  bool res = decompose();
  if(!res)
    return false;

  // 利用重投影误差得到满足误差范围的匹配点   是否匹配保存在inliers bool数组中
  computeMatchesInliers();

  findBestDecomposition();
  T_c2_from_c1 = decompositions.front().T;
  return true;
}


// 从单应矩阵恢复R t
bool Homography::
decompose()
{
  decompositions.clear();
  JacobiSVD<MatrixXd> svd(H_c2_from_c1, ComputeThinU | ComputeThinV);

  // 对单应矩阵进行svd分解  得到奇异值
  Vector3d singular_values = svd.singularValues();

  double d1 = fabs(singular_values[0]); // The paper suggests the square of these (e.g. the evalues of AAT)
  double d2 = fabs(singular_values[1]); // should be used, but this is wrong. c.f. Faugeras' book.
  double d3 = fabs(singular_values[2]);

  // 得到奇异值分解中的U和V
  Matrix3d U = svd.matrixU();
  Matrix3d V = svd.matrixV();                    // VT^T

  // UV行列式的积
  double s = U.determinant() * V.determinant();

  double dPrime_PM = d2;

  int nCase;
  if(d1 != d2 && d2 != d3)
    nCase = 1;
  else if( d1 == d2 && d2 == d3)
    nCase = 3;
  else
    nCase = 2;

  if(nCase != 1)
  {
    printf("FATAL Homography Initialization: This motion case is not implemented or is degenerate. Try again. ");
    return false;
  }

  double x1_PM;
  double x2;
  double x3_PM;

  // All below deals with the case = 1 case.
  // Case 1 implies (d1 != d3)
  { // Eq. 12
    x1_PM = sqrt((d1*d1 - d2*d2) / (d1*d1 - d3*d3));
    x2    = 0;
    x3_PM = sqrt((d2*d2 - d3*d3) / (d1*d1 - d3*d3));
  };

  double e1[4] = {1.0,-1.0, 1.0,-1.0};
  double e3[4] = {1.0, 1.0,-1.0,-1.0};

  Vector3d np;
  HomographyDecomposition decomp;

  // Case 1, d' > 0:
  decomp.d = s * dPrime_PM;
  for(size_t signs=0; signs<4; signs++)
  {
    // Eq 13
    decomp.R = Matrix3d::Identity();
    double dSinTheta = (d1 - d3) * x1_PM * x3_PM * e1[signs] * e3[signs] / d2;
    double dCosTheta = (d1 * x3_PM * x3_PM + d3 * x1_PM * x1_PM) / d2;
    decomp.R(0,0) = dCosTheta;
    decomp.R(0,2) = -dSinTheta;
    decomp.R(2,0) = dSinTheta;
    decomp.R(2,2) = dCosTheta;

    // Eq 14
    decomp.t[0] = (d1 - d3) * x1_PM * e1[signs];
    decomp.t[1] = 0.0;
    decomp.t[2] = (d1 - d3) * -x3_PM * e3[signs];

    np[0] = x1_PM * e1[signs];
    np[1] = x2;
    np[2] = x3_PM * e3[signs];
    decomp.n = V * np;

    decompositions.push_back(decomp);
  }

  // Case 1, d' < 0:
  decomp.d = s * -dPrime_PM;
  for(size_t signs=0; signs<4; signs++)
  {
    // Eq 15
    decomp.R = -1 * Matrix3d::Identity();
    double dSinPhi = (d1 + d3) * x1_PM * x3_PM * e1[signs] * e3[signs] / d2;
    double dCosPhi = (d3 * x1_PM * x1_PM - d1 * x3_PM * x3_PM) / d2;
    decomp.R(0,0) = dCosPhi;
    decomp.R(0,2) = dSinPhi;
    decomp.R(2,0) = dSinPhi;
    decomp.R(2,2) = -dCosPhi;

    // Eq 16
    decomp.t[0] = (d1 + d3) * x1_PM * e1[signs];
    decomp.t[1] = 0.0;
    decomp.t[2] = (d1 + d3) * x3_PM * e3[signs];

    np[0] = x1_PM * e1[signs];
    np[1] = x2;
    np[2] = x3_PM * e3[signs];
    decomp.n = V * np;

    decompositions.push_back(decomp);
  }

  // Save rotation and translation of the decomposition
  for(unsigned int i=0; i<decompositions.size(); i++)
  {
    Matrix3d R = s * U * decompositions[i].R * V.transpose();
    Vector3d t = U * decompositions[i].t;
    decompositions[i].T = Sophus::SE3(R, t);
  }
  return true;
}

bool operator<(const HomographyDecomposition lhs, const HomographyDecomposition rhs)
{
  return lhs.score < rhs.score;
}

void Homography::
findBestDecomposition()
{
  assert(decompositions.size() == 8);
  // 遍历所有可能的R t分解
  
  
  // 判断地图点在相机的正面还是背面
  for(size_t i=0; i<decompositions.size(); i++)
  {
    HomographyDecomposition &decom = decompositions[i];
    size_t nPositive = 0;
    // 对于一组r t分解，遍历所有的匹配内点
    for(size_t m=0; m<fts_c1.size(); m++)
    {
      if(!inliers[m])
        continue;
      
      // 参考帧角点的归一化坐标
      const Vector2d& v2 = fts_c1[m];
      
      // ??????????????????????????这里的decom.d是什么
      // 把归一化uv 转换成uv1乘以R的第三行 得到Z??????????????????????????
      double dVisibilityTest = (H_c2_from_c1(2,0) * v2[0] + H_c2_from_c1(2,1) * v2[1] + H_c2_from_c1(2,2)) / decom.d;
      if(dVisibilityTest > 0.0)
        nPositive++;
    }
    decom.score = -nPositive;
  }

  sort(decompositions.begin(), decompositions.end());
  decompositions.resize(4);


  // ??????????????????????????????
  for(size_t i=0; i<decompositions.size(); i++)
  {
    HomographyDecomposition &decom = decompositions[i];
    int nPositive = 0;
    for(size_t m=0; m<fts_c1.size(); m++)
    {
      if(!inliers[m])
        continue;

      // X/Z,Y/Z->  X/Z,Y/Z,1
      Vector3d v3 = unproject2d(fts_c1[m]);
      double dVisibilityTest = v3.dot(decom.n) / decom.d;
      if(dVisibilityTest > 0.0)
        nPositive++;
    };
    decom.score = -nPositive;
  }

  sort(decompositions.begin(), decompositions.end());
  decompositions.resize(2);


  // According to Faugeras and Lustman, ambiguity exists if the two scores are equal
  // but in practive, better to look at the ratio!
  double dRatio = (double) decompositions[1].score / (double) decompositions[0].score;

  // 如果一组r t分解的得分比第二名高很多，则可以直接返回
  if(dRatio < 0.9) // no ambiguity!
    decompositions.erase(decompositions.begin() + 1);

  // 如果第一名rt分解没有比第二名高很多
  else  // two-way ambiguity. Resolve by sampsonus score of all points.
  {
    double dErrorSquaredLimit  = thresh * thresh * 4;
    // 对于前两名  分别计算一个误差值   选择误差值小的那一个作为Rt分解的值
    double adSampsonusScores[2];
    for(size_t i=0; i<2; i++)
    {
      Sophus::SE3 T = decompositions[i].T;
      Matrix3d Essential = T.rotation_matrix() * sqew(T.translation());
      double dSumError = 0;

      // 遍历所有的匹配角点
      for(size_t m=0; m < fts_c1.size(); m++ )
      {
        // 在math_utils.cpp中  根据匹配点和单应矩阵 得到误差
        // ??????????????没看懂....
        double d = sampsonusError(fts_c1[m], Essential, fts_c2[m]);
        if(d > dErrorSquaredLimit)
          d = dErrorSquaredLimit;
        dSumError += d;
      }
      adSampsonusScores[i] = dSumError;
    }

    if(adSampsonusScores[0] <= adSampsonusScores[1])
      decompositions.erase(decompositions.begin() + 1);
    else
      decompositions.erase(decompositions.begin());
  }
}


} /* end namespace vk */
