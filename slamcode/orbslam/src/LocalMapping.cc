/**
* This file is part of ORB-SLAM.
*
* Copyright (C) 2014 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <http://webdiis.unizar.es/~raulmur/orbslam/>
*
* ORB-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include <ros/ros.h>

namespace ORB_SLAM
{

LocalMapping::LocalMapping(Map *pMap):
    mbResetRequested(false), mpMap(pMap),  mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    ros::Rate r(500);
    while(ros::ok())
    {
        // Check if there are keyframes in the queue
        // 如果tracking插入了新的关键帧
        // 只有当加入了关键帧的情况下才进行全局map的更新
        if(CheckNewKeyFrames())
        {  
                  
            // Tracking will see that Local Mapping is busy
            // local mapping开始工作  不再接受新的关键帧
            SetAcceptKeyFrames(false);


            // BoW conversion and insertion in Map
            // 处理新的关键帧 更新共视图等操作
            ProcessNewKeyFrame();

            // Check recent MapPoints
            MapPointCulling();

            // Triangulate new MapPoints
            // 通过三角化 形成新的地图点
            CreateNewMapPoints();


            // Find more matches in neighbor keyframes and fuse point duplications
            // 对于地图点  在当前帧和邻居关键帧中分别寻找最优匹配点  如果已经有匹配 则进行融合
            SearchInNeighbors();

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                // 进行局部BA  形成局部关键帧和对应的地图点  有一部分关键帧固定   两轮优化  第一轮优化 删除离群点  最后得到优化后关键帧的位姿和地图点的三维坐标
                Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA);

                // Check redundant local Keyframes
                // 剔除关键帧
                KeyFrameCulling();


                // updateflag = true
                mpMap->SetFlagAfterBA();

                // Tracking will see Local Mapping idle
                if(!CheckNewKeyFrames())
                    SetAcceptKeyFrames(true);
            }
            //mlpLoopKeyFrameQueue.pushback
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }


        // Safe area to stop
        if(stopRequested())
        {
            Stop();
            ros::Rate r2(1000);
            while(isStopped() && ros::ok())
            {
                r2.sleep();
            }

            SetAcceptKeyFrames(true);
        }


        ResetIfRequested();
        r.sleep();
    }
}


//插入关键帧  mlNewKeyFrames vector
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
    SetAcceptKeyFrames(false);
}


bool LocalMapping::CheckNewKeyFrames()
{
    boost::mutex::scoped_lock lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}



//处理tracing传进来的新的关键帧
void LocalMapping::ProcessNewKeyFrame()
{
    {
        boost::mutex::scoped_lock lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }



    // Compute Bags of Words structures
    // 计算该关键帧的词袋
    mpCurrentKeyFrame->ComputeBoW();


    if(mpCurrentKeyFrame->mnId==0)
        return;



    // Associate MapPoints to the new keyframe and update normal and descriptor
   
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    if(mpCurrentKeyFrame->mnId>1) //This operations are already done in the tracking for the first two keyframes
    {
        // 遍历当前关键帧关联的地图点
        for(size_t i=0; i<vpMapPointMatches.size(); i++)
        {
        
            MapPoint* pMP = vpMapPointMatches[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    // 将该关键帧和对应的像素索引添加到地图点可视变量中
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
            }
        }
    }

    if(mpCurrentKeyFrame->mnId==1)
    {
        for(size_t i=0; i<vpMapPointMatches.size(); i++)
        {
            MapPoint* pMP = vpMapPointMatches[i];
            if(pMP)
            {
                mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }  

    // Update links in the Covisibility Graph
    // 更新共视图
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map  mspKeyFrames.insert
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}



// 对当前所有的地图点进行检查
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    // mlpRecentAddedMapPoints保存新加入的地图点
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;
    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            //设置flag   将该地图点删除
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if((nCurrentKFid-pMP->mnFirstKFid)>=2 && pMP->Observations()<=2)
        {

            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if((nCurrentKFid-pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}



void LocalMapping::CreateNewMapPoints()
{
    // Take neighbor keyframes in covisibility graph
    // 得到与当前关键帧共视点最多的20个关键帧
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(20);

    ORBmatcher matcher(0.6,false);


    // 得到当前关键帧的位姿和相机光心
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();


    //相机内参
    const float fx1 = mpCurrentKeyFrame->fx;
    const float fy1 = mpCurrentKeyFrame->fy;
    const float cx1 = mpCurrentKeyFrame->cx;
    const float cy1 = mpCurrentKeyFrame->cy;
    const float invfx1 = 1.0f/fx1;
    const float invfy1 = 1.0f/fy1;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->GetScaleFactor();




    // Search matches with epipolar restriction and triangulate
    // 遍历所有的候选关键帧
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        // Small translation errors for short baseline keyframes make scale to diverge
        // 得到候选关键帧的光心
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        //基线
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);
        
        // 得到候选关键帧关联地图点的平均深度值
        const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
        const float ratioBaselineDepth = baseline/medianDepthKF2;


        if(ratioBaselineDepth<0.01)
            continue;



        // Compute Fundamental Matrix
        // 根据两个关键帧的位姿  计算基础矩阵	就可以得到两帧对应像素之间的关系
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);


        // Search matches that fulfil epipolar constraint
        vector<cv::KeyPoint> vMatchedKeysUn1;
        vector<cv::KeyPoint> vMatchedKeysUn2;
        vector<pair<size_t,size_t> > vMatchedIndices;
        
        
        //  对于当前关键帧和参考关键帧中 寻找没有与地图点相关联的关键点之间的匹配  匹配的点  
		// 匹配上的 当前关键帧中的关键点索引存入vMatchedKeysUn1  参考帧中的关键点索引存入vMatchedKeysUn2  vMatchedIndices存储两个的pair对
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedKeysUn1,vMatchedKeysUn2,vMatchedIndices);


       //参考帧的位姿
        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        //参考帧的相机内参
        const float fx2 = pKF2->fx;
        const float fy2 = pKF2->fy;
        const float cx2 = pKF2->cx;
        const float cy2 = pKF2->cy;
        const float invfx2 = 1.0f/fx2;
        const float invfy2 = 1.0f/fy2;

        // Triangulate each match
        for(size_t ikp=0, iendkp=vMatchedKeysUn1.size(); ikp<iendkp; ikp++)
        {
            const int idx1 = vMatchedIndices[ikp].first;
            const int idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = vMatchedKeysUn1[ikp];
            const cv::KeyPoint &kp2 = vMatchedKeysUn2[ikp];

            // Check parallax between rays
            // xn1，xn2是归一化坐标
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0 );
            cv::Mat ray1 = Rwc1*xn1;

            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0 );
            cv::Mat ray2 = Rwc2*xn2;

            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            if(cosParallaxRays<0 || cosParallaxRays>0.9998)
                continue;

            // Linear Triangulation Method
            // 如果有乘以K，需要使用像素坐标，否则需要使用归一化坐标，就像这里一样
            cv::Mat A(4,4,CV_32F);
            A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
            A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
            A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
            A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

            cv::Mat w,u,vt;
            // svd分解
            cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

			// 得到相对于第一帧的地图点三维坐标  齐次坐标
            cv::Mat x3D = vt.row(3).t();

            if(x3D.at<float>(3)==0)
                continue;

            // Euclidean coordinates
			// 得到地图点的三维坐标
            x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
            cv::Mat x3Dt = x3D.t();

			// 投影到当前帧  判断深度
            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;
			// 投影到参考帧 判断深度
            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;


            // 根据R和T进行重投影 (此处三维点是不准确的)
            //Check reprojection error in first keyframe
            float sigmaSquare1 = mpCurrentKeyFrame->GetSigma2(kp1.octave);
            float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            float invz1 = 1.0/z1;
            float u1 = fx1*x1*invz1+cx1;
            float v1 = fy1*y1*invz1+cy1;
            float errX1 = u1 - kp1.pt.x;
            float errY1 = v1 - kp1.pt.y;
            if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                continue;

            //Check reprojection error in second keyframe
            float sigmaSquare2 = pKF2->GetSigma2(kp2.octave);
            float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            float invz2 = 1.0/z2;
            float u2 = fx2*x2*invz2+cx2;
            float v2 = fy2*y2*invz2+cy2;
            float errX2 = u2 - kp2.pt.x;
            float errY2 = v2 - kp2.pt.y;
            if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                continue;




            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            float ratioDist = dist1/dist2;
            float ratioOctave = mpCurrentKeyFrame->GetScaleFactor(kp1.octave)/pKF2->GetScaleFactor(kp2.octave);
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            pMP->AddObservation(pKF2,idx2);
            pMP->AddObservation(mpCurrentKeyFrame,idx1);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            // 将该地图点设置为新加入的地图点
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }
}





void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    // 获得当前帧共视点最多的20个关键帧
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(20);
    // 筛选的候选关键帧
    vector<KeyFrame*> vpTargetKFs;
    
    
    // 得到一系列候选关键帧  vpTargetKFs
    for(vector<KeyFrame*>::iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;


        // 查找邻居的邻居
        // Extend to some second neighbors
        vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher(0.6);
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        // 遍历候选关键帧  对于当前帧关联的地图点，在候选关键帧中寻找最佳匹配
        //  候选关键帧  当前关键帧关联的地图点
        matcher.Fuse(pKFi,vpMapPointMatches);
    }



    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());
    // 遍历所有候选关键帧
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();
        // 对于候选关键帧关联的地图点
        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }
    
    //  对于候选关键帧关联的地图点 在当前帧中寻找最优匹配
    // 当前帧  候选关键帧关联的地图点
    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}


// 当前帧  参考关键帧
// 相当于有了两帧的位姿，可以计算得到两帧之间的基础矩阵，就可以得到两帧对应像素坐标之间的关系
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    cv::Mat K1 = pKF1->GetCalibrationMatrix();
    cv::Mat K2 = pKF2->GetCalibrationMatrix();


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    boost::mutex::scoped_lock lock(mMutexStop);
    mbStopRequested = true;
    boost::mutex::scoped_lock lock2(mMutexNewKFs);
    mbAbortBA = true;
}

void LocalMapping::Stop()
{
    boost::mutex::scoped_lock lock(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isStopped()
{
    boost::mutex::scoped_lock lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    boost::mutex::scoped_lock lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    boost::mutex::scoped_lock lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();
}

bool LocalMapping::AcceptKeyFrames()
{
    boost::mutex::scoped_lock lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

//设置mbAcceptKeyFrames的值
void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    boost::mutex::scoped_lock lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}


// 关键帧剔除
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // 获得当前帧具有最多共视点的关键帧
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    // 遍历这些关键帧
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
    
        // 判断关键帧pKF是否需要删除
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        
        // 得到待判断关键帧关联的地图点
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        // 能够被其他三个关键帧看到的地图点数目
        int nRedundantObservations=0;
        // 待判断关键帧关联的有效地图点的数目
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    nMPs++;
                    // 如果地图点的共视关键帧大于3
                    if(pMP->Observations()>3)
                    {
                        int scaleLevel = pKF->GetKeyPointUn(i).octave;
                        map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        
                        
                        
                        // 遍历所有共视关键帧
                        for(map<KeyFrame*, size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            int scaleLeveli = pKFi->GetKeyPointUn(mit->second).octave;
                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=3)
                                    break;
                            }
                        }
                        if(nObs>=3)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
                                  v.at<float>(2),               0,-v.at<float>(0),
                                 -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        boost::mutex::scoped_lock lock(mMutexReset);
        mbResetRequested = true;
    }

    ros::Rate r(500);
    while(ros::ok())
    {
        {
        boost::mutex::scoped_lock lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        r.sleep();
    }
}

void LocalMapping::ResetIfRequested()
{
    boost::mutex::scoped_lock lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

} //namespace ORB_SLAM
