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
#include <stdexcept>
#include <svo/reprojector.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/map.h>
#include <svo/config.h>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <vikit/abstract_camera.h>
#include <vikit/math_utils.h>
#include <vikit/timer.h>

namespace svo {

Reprojector::Reprojector(vk::AbstractCamera* cam, Map& map) :
    map_(map)
{
  initializeGrid(cam);
}

Reprojector::~Reprojector()
{
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ delete c; });
}

void Reprojector::initializeGrid(vk::AbstractCamera* cam)
{
  grid_.cell_size = Config::gridSize();
  grid_.grid_n_cols = ceil(static_cast<double>(cam->width())/grid_.cell_size);
  grid_.grid_n_rows = ceil(static_cast<double>(cam->height())/grid_.cell_size);
  grid_.cells.resize(grid_.grid_n_cols*grid_.grid_n_rows);
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell*& c){ c = new Cell; });
  grid_.cell_order.resize(grid_.cells.size());
  for(size_t i=0; i<grid_.cells.size(); ++i)
    grid_.cell_order[i] = i;
  random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end()); // maybe we should do it at every iteration!
}

void Reprojector::resetGrid()
{
  n_matches_ = 0;
  n_trials_ = 0;
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ c->clear(); });
}


// 当前帧  overlap_kfs存的是与当前帧有共视点的其他关键帧  和共视角点的数量
void Reprojector::reprojectMap(
    FramePtr frame,
    std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs)
{
  resetGrid();

  // Identify those Keyframes which share a common field of view.
  SVO_START_TIMER("reproject_kfs");
  list< pair<FramePtr,double> > close_kfs;
  
  // 对于当前帧  根据每一帧的五个关键角点 找到有共视点的其他关键帧，存储关键帧 和该关键帧与当前帧的距离
  map_.getCloseKeyframes(frame, close_kfs);

  // Sort KFs with overlap according to their closeness
  close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                 boost::bind(&std::pair<FramePtr, double>::second, _2));

  // Reproject all mappoints of the closest N kfs with overlap. We only store
  // in which grid cell the points fall.
  size_t n = 0;
  // 10个
  overlap_kfs.reserve(options_.max_n_kfs);
  
  
  // 遍历所有的共视关键帧
  for(auto it_frame=close_kfs.begin(), ite_frame=close_kfs.end();
      it_frame!=ite_frame && n<options_.max_n_kfs; ++it_frame, ++n)
  {
    FramePtr ref_frame = it_frame->first;
    overlap_kfs.push_back(pair<FramePtr,size_t>(ref_frame,0));

    // Try to reproject each mappoint that the other KF observes
    // 遍历该关键帧所有的角点
    for(auto it_ftr=ref_frame->fts_.begin(), ite_ftr=ref_frame->fts_.end();
        it_ftr!=ite_ftr; ++it_ftr)
    {
      // check if the feature has a mappoint assigned
      if((*it_ftr)->point == NULL)
        continue;

      // make sure we project a point only once
      if((*it_ftr)->point->last_projected_kf_id_ == frame->id_)
        continue;
      (*it_ftr)->point->last_projected_kf_id_ = frame->id_;
      // 如果共视关键帧角点对应的地图点能投影在当前帧  索引数量加1
      // reprojectPoint 如果投影到当前帧 像素点能取得8*8的图块，就把地图点存入当前帧投影位置的网格中
      if(reprojectPoint(frame, (*it_ftr)->point))
        overlap_kfs.back().second++;
    }
  }
  SVO_STOP_TIMER("reproject_kfs");



  // Now project all point candidates
  SVO_START_TIMER("reproject_candidates");
  {
    boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);
    
    // 遍历地图上的候选地图点???????????????怎么初始化的?????????????????
    // candidates_是point+feature的列表
    auto it=map_.point_candidates_.candidates_.begin();
    while(it!=map_.point_candidates_.candidates_.end())
    {
    	// 将候选地图点向当前帧投影  如果能取到8*8的图块,存入网格
    	// 如果地图点有10次投影不成功，删除该候选点
    	// 将该地图点从候选地图点中删除
      if(!reprojectPoint(frame, it->first))
      {
        it->first->n_failed_reproj_ += 3;
        if(it->first->n_failed_reproj_ > 30)
        {
        	// map.cpp 放入trash_points_队列
          map_.point_candidates_.deleteCandidate(*it);
          it = map_.point_candidates_.candidates_.erase(it);
          continue;
        }
      }
      ++it;
    }
  } // unlock the mutex when out of scope
  SVO_STOP_TIMER("reproject_candidates");




  // Now we go through each grid cell and select one point to match.
  // At the end, we should have at maximum one reprojected point per cell.
  SVO_START_TIMER("feature_align");
  // 遍历网格 
  for(size_t i=0; i<grid_.cells.size(); ++i)
  {
    // we prefer good quality points over unkown quality (more likely to match)
    // and unknown quality over candidates (position not optimized)
    // 将网格中的地图点进行排序  good> undown>candidate>deleted
    // 所以每个cell只选择一个质量最好的特征点  而且选择了maxFts个cell就停止了
    // grid_.cell_order打乱了cell的顺序
    // 随机对cell中最好的地图点进行投影  成功投影后  n_matches_加1
    // reprojectCell 随机选择一个cell中最好的地图点，优化在当前帧中的像素位置  成为当前帧的特征点
    if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame))
      ++n_matches_;
    // 120
    if(n_matches_ > (size_t) Config::maxFts())
      break;
  }
  SVO_STOP_TIMER("feature_align");
}

bool Reprojector::pointQualityComparator(Candidate& lhs, Candidate& rhs)
{
	// pt是地图点
  if(lhs.pt->type_ > rhs.pt->type_)
    return true;
  return false;
}


// 当前帧的cell  当前帧
// cell 在reprojector.h中
// cell->pt是参考关键帧特征点对应的地图点  cell->px是该地图点通过当前帧的位姿投影到当前帧的像素坐标
bool Reprojector::reprojectCell(Cell& cell, FramePtr frame)
{
	// 将网格中的地图点进行排序  good> undown>candidate>deleted
  cell.sort(boost::bind(&Reprojector::pointQualityComparator, _1, _2));
  Cell::iterator it=cell.begin();
  while(it!=cell.end())
  {
    ++n_trials_;

    // 在网格中删除type为deleted的地图点
    if(it->pt->type_ == Point::TYPE_DELETED)
    {
      it = cell.erase(it);
      continue;
    }

    bool found_match = true;
    // true
    if(options_.find_match_direct)
    	// matcher.cpp中  
    	// patch_ 和patch_with_border都是个啥东西 在哪里初始化的 。。。??????????????????????
    	// findMatchDirect函数返回后 it->px 为优化后的地图点对应当前帧的像素坐标
      found_match = matcher_.findMatchDirect(*it->pt, *frame, it->px);
    
    if(!found_match)
    {
      it->pt->n_failed_reproj_++;
      if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_failed_reproj_ > 15)
        map_.safeDeletePoint(it->pt);
      if(it->pt->type_ == Point::TYPE_CANDIDATE  && it->pt->n_failed_reproj_ > 30)
        map_.point_candidates_.deleteCandidatePoint(it->pt);
      it = cell.erase(it);
      continue;
    }
    
    it->pt->n_succeeded_reproj_++;
    if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_succeeded_reproj_ > 10)
      it->pt->type_ = Point::TYPE_GOOD;

    // feature.h中  为得到的像素坐标初始化角点特征
    // 在当前帧中得到新的特征点，注意这里的level是前面得到的最优尺度
    Feature* new_feature = new Feature(frame.get(), it->px, matcher_.search_level_);
    // 将特征加入fts_中
    frame->addFeature(new_feature);

    // Here we add a reference in the feature to the 3D point, the other way
    // round is only done if this frame is selected as keyframe.
    // 将当前帧像素点和地图点建立联系
    new_feature->point = it->pt;

    // ref_ftr_：做图像块匹配的关键帧
    if(matcher_.ref_ftr_->type == Feature::EDGELET)
    {
      new_feature->type = Feature::EDGELET;
      // A_cur_ref_为从参考帧到当前帧的仿射变换 * 参考帧对应像素点的梯度  成为当前帧像素点的梯度
      new_feature->grad = matcher_.A_cur_ref_*matcher_.ref_ftr_->grad;
      new_feature->grad.normalize();
    }

    // If the keyframe is selected and we reproject the rest, we don't have to
    // check this point anymore.
    it = cell.erase(it);

    // Maximum one point per cell.
    return true;
  }
  return false;
}


// 当前帧  待投影的地图点
bool Reprojector::reprojectPoint(FramePtr frame, Point* point)
{
	// 先将地图点利用当前帧的位姿 转换到当前坐标系  然后变成X/Z,Y/Z,接着转换为像素坐标
  Vector2d px(frame->w2c(point->pos_));
  // abstract_camera.h  是否在8~width-8,8~height-8范围内
  // 如果投影到当前帧的像素点处 可以取得8*8的图块 
  // ?????????????????????这里的patch size in matcher是什么
  if(frame->cam_->isInFrame(px.cast<int>(), 8)) // 8px is the patch size in the matcher
  {
    const int k = static_cast<int>(px[1]/grid_.cell_size)*grid_.grid_n_cols
                + static_cast<int>(px[0]/grid_.cell_size);
    // Candidate 构造函数  在reprojector.h中
    grid_.cells.at(k)->push_back(Candidate(point, px));
    return true;
  }
  return false;
}

} // namespace svo
