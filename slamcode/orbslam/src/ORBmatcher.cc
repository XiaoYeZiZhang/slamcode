/**
* This file is part of ORB-SLAM.
* It is based on the file orb.cpp from the OpenCV library (see BSD license below)
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

#include "ORBmatcher.h"

#include<limits.h>

#include<ros/ros.h>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>


using namespace std;

namespace ORB_SLAM
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;


ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}



int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches=0;

    const bool bFactor = th!=1.0;

    //遍历vpMapPoints的地图点
    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        // 遍历局部地图点中可以在当前帧中投影的  在uv区域内搜索与三维坐标对应的最佳像素坐标
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView)
            continue;

        if(pMP->isBad())
            continue;

        const int &nPredictedLevel = pMP->mnTrackScaleLevel;    

        // The size of the window will depend on the viewing direction
        // 2.5或4.0
        float r = RadiusByViewingCos(pMP->mTrackViewCos);


        if(bFactor)
            r*=th;

        vector<size_t> vNearIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vNearIndices.empty())
            continue;

        cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist=INT_MAX;
        int bestLevel= -1;
        int bestDist2=INT_MAX;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        for(vector<size_t>::iterator vit=vNearIndices.begin(), vend=vNearIndices.end(); vit!=vend; vit++)
        {
            size_t idx = *vit;

            if(F.mvpMapPoints[idx])
                continue;

            cv::Mat d=F.mDescriptors.row(idx);

            const int dist = DescriptorDistance(MPdescriptor,d);

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist<=TH_HIGH)
        {
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;

            F.mvpMapPoints[bestIdx]=pMP;
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}



// 当前帧中的关键点    参考帧中的关键点   从参考帧到当前帧的基础矩阵   参考帧
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    // 又来???? 这里应该是F12的逆???? 又变成转置
	// 利用当前帧的像素  和 从当前帧到参考帧的基础矩阵(F12的逆)  得到在参考帧中对应的像素坐标
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;


    // 极线约束
    if(den==0)
        return false;

    const float dsqr = num*num/den;

    return dsqr<3.84*pKF2->GetSigma2(kp2.octave);
}



//matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i])


//matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);
// 根据词袋进行搜索
// ref:https://blog.csdn.net/qq_30356613/article/details/80587889
// 首先取出关键帧和当前帧的特征向量（注意这里的特征向量是根据节点id排序好的特征向量（用DBOW库中的特征向量），便于我们进行搜索），
//然后遍历关键帧的每一个特征向量，在当前帧的同一节点下搜索其匹配点（计算同一节点特征向量对应描述子的距离最小匹配作为其匹配点）。
// 最后进行匹配筛选（剔除误匹配）

int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    //获得候选关键帧的地图点mvpMapPoints
    vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();


    vpMapPointMatches = vector<MapPoint*>(F.mvpMapPoints.size(),static_cast<MapPoint*>(NULL));

    //获得候选关键帧的特征向量
    DBoW2::FeatureVector vFeatVecKF = pKF->GetFeatureVector();

    int nmatches=0;

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::iterator Fit = F.mFeatVec.begin();
    
    DBoW2::FeatureVector::iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::iterator Fend = F.mFeatVec.end();


    while(KFit != KFend && Fit != Fend)
    {
        // 同一个节点上
        if(KFit->first == Fit->first)
        {
            vector<unsigned int> vIndicesKF = KFit->second;
            vector<unsigned int> vIndicesF = Fit->second;

            //遍历候选关键帧特征向量的second 难道是一系列关键点的索引??????????????  featureVector到底是什么
            for(size_t iKF=0, iendKF=vIndicesKF.size(); iKF<iendKF; iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];
                //得到地图点
                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;

                //得到关键点的描述子
                cv::Mat dKF= pKF->GetDescriptor(realIdxKF);


                int bestDist1=INT_MAX;
                int bestIdxF =-1 ;
                int bestDist2=INT_MAX;

                //遍历所有当前帧的特征向量second   根据描述子的距离 得到两个与候选关键帧的最优匹配
                for(size_t iF=0, iendF=vIndicesF.size(); iF<iendF; iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];
                    
                    
                    //对还没有与mapPoint匹配的点进行操作
                    if(vpMapPointMatches[realIdxF])
                        continue;

                    //得到描述子
                    cv::Mat dF = F.mDescriptors.row(realIdxF).clone();

                    const int dist =  DescriptorDistance(dKF,dF);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }



                if(bestDist1<=TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        //将当前帧匹配上的点与 地图点建立联系
                        vpMapPointMatches[bestIdxF]=pMP;
                        //得到候选关键帧对应的关键点
                        cv::KeyPoint kp = pKF->GetKeyPointUn(realIdxKF);

                        if(mbCheckOrientation)
                        {
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }
                }

            }

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }


    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=NULL;
                nmatches--;
            }
        }
    }

    return nmatches;
}


// matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float fx = pKF->fx;
    const float fy = pKF->fy;
    const float cx = pKF->cx;
    const float cy = pKF->cy;

    const int nMaxLevel = pKF->GetScaleLevels()-1;
    vector<float> vfScaleFactors = pKF->GetScaleFactors();

    // 由相似矩阵×T 变换为Rcw tcw
    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));

    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(NULL);

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    // 遍历回环关键帧组所有的地图点
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        //重投影到当前帧
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        // Compute predicted scale level
        const float ratio = dist/minDistance;

        vector<float>::iterator it = lower_bound(vfScaleFactors.begin(), vfScaleFactors.end(), ratio);
        const int nPredictedLevel = min(static_cast<int>(it-vfScaleFactors.begin()),nMaxLevel);

        // Search in a radius
        const float radius = th*pKF->GetScaleFactor(nPredictedLevel);
        // 确定搜索区域
        vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int kpLevel= pKF->GetKeyPointScaleLevel(idx);

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            cv::Mat dKF = pKF->GetDescriptor(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }

        }

        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}

//matcher.WindowSearch(mLastFrame,mCurrentFrame,200,vpMapPointMatches,minOctave);
//在窗口中寻找参考帧对应地图点的像素，得到最优匹配
int ORBmatcher::WindowSearch(Frame &F1, Frame &F2, int windowSize, vector<MapPoint *> &vpMapPointMatches2, int minScaleLevel, int maxScaleLevel)
{
    int nmatches=0;
    //mvpMapPoints是该帧关键点索引与地图点的对应关系
    vpMapPointMatches2 = vector<MapPoint*>(F2.mvpMapPoints.size(),static_cast<MapPoint*>(NULL));
    //vnMatches21初始化为去畸变后关键点的数目
    vector<int> vnMatches21 = vector<int>(F2.mvKeysUn.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const bool bMinLevel = minScaleLevel>0;
    const bool bMaxLevel= maxScaleLevel<INT_MAX;


    //遍历参考帧的地图点
    for(size_t i1=0, iend1=F1.mvpMapPoints.size(); i1<iend1; i1++)
    {
    
        MapPoint* pMP1 = F1.mvpMapPoints[i1];

        if(!pMP1)
            continue;
        if(pMP1->isBad())
            continue;

        //参考帧地图点对应的关键点kp1
        const cv::KeyPoint &kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;

        if(bMinLevel)
            if(level1<minScaleLevel)
                continue;

        if(bMaxLevel)
            if(level1>maxScaleLevel)
                continue;

        //在参考帧该关键点周围的窗口中返回符合条件的关键点索引
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(kp1.pt.x,kp1.pt.y, windowSize, level1, level1);

        if(vIndices2.empty())
            continue;


        //参考帧中该关键点的描述子
        cv::Mat d1 = F1.mDescriptors.row(i1);
        
        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        //遍历当前帧所有可能是关键点的位置  根据描述子的距离得到最有可能的两个匹配
        for(vector<size_t>::iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
        {
            size_t i2 = *vit;

            if(vpMapPointMatches2[i2])
                continue;

            //获得候选位置的描述子
            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            } else if(dist<bestDist2)
            {
                bestDist2=dist;
            }                                        
        }



        if(bestDist<=bestDist2*mfNNratio && bestDist<=TH_HIGH)
        {
            //得到最优匹配的关键点索引bestIdx2,和地图点建立联系
            vpMapPointMatches2[bestIdx2]=pMP1;
            vnMatches21[bestIdx2]=i1;
            nmatches++;

            float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
            if(rot<0.0)
                rot+=360.0f;
            int bin = round(rot*factor);
            if(bin==HISTO_LENGTH)
                bin=0;
            ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
            rotHist[bin].push_back(bestIdx2);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vpMapPointMatches2[rotHist[i][j]]=NULL;
                    vnMatches21[rotHist[i][j]]=-1;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}





//参考帧 当前帧  窗口大小  从参考帧对应地图点与当前帧的关键点的对应
int ORBmatcher::SearchByProjection(Frame &F1, Frame &F2, int windowSize, vector<MapPoint *> &vpMapPointMatches2)
{
    //当前帧  经过初始位姿估计删选得到的地图点与关键点的对应
    vpMapPointMatches2 = F2.mvpMapPoints;
    //筛了一部分的地图点为spMapPointsAlreadyFound
    set<MapPoint*> spMapPointsAlreadyFound(vpMapPointMatches2.begin(),vpMapPointMatches2.end());



    int nmatches = 0;
    const cv::Mat Rc2w = F2.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tc2w = F2.mTcw.rowRange(0,3).col(3);

    //遍历所有的地图点(不经删选 因为是F1对应的)   对于前面BA删除掉的对应 经过计算得到的Rt进行重投影，在uv的一个区域寻找与参考帧关键点的最优匹配
    //如果满足一定条件，则将找到的最优位置与地图点匹配
    for(size_t i1=0, iend1=F1.mvpMapPoints.size(); i1<iend1; i1++)
    {
        MapPoint* pMP1 = F1.mvpMapPoints[i1];

        if(!pMP1)
            continue;
        //如果已经有了mapPoint的对应，直接跳过，也就是只寻找离群点的mapPoint对应的正确的像素点
        if(pMP1->isBad() || spMapPointsAlreadyFound.count(pMP1))
            continue;

        //参考帧对应的关键点
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        //地图点在第一帧坐标系下的坐标
        cv::Mat x3Dw = pMP1->GetWorldPos();
        //地图点在当前帧坐标系下的坐标
        cv::Mat x3Dc2 = Rc2w*x3Dw+tc2w;

        //有了位姿  得到地图点在当前坐标系下的坐标  再通过相机内参 得到像素坐标
        const float xc2 = x3Dc2.at<float>(0);
        const float yc2 = x3Dc2.at<float>(1);
        const float invzc2 = 1.0/x3Dc2.at<float>(2);

        float u2 = F2.fx*xc2*invzc2+F2.cx;
        float v2 = F2.fy*yc2*invzc2+F2.cy;
        //得到一个搜索区域
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(u2,v2, windowSize, level1, level1);

        if(vIndices2.empty())
            continue;



        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;


        for(vector<size_t>::iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
        {
            size_t i2 = *vit;

            if(vpMapPointMatches2[i2])
                continue;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            } else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        if(static_cast<float>(bestDist)<=static_cast<float>(bestDist2)*mfNNratio && bestDist<=TH_HIGH)
        {
            vpMapPointMatches2[bestIdx2]=pMP1;
            nmatches++;
        }

    }

    return nmatches;
}


// Matching for the Map Initialization
// int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10);
//对于F1中的每一个关键点，在一个窗口范围寻找F2中与之匹配的关键点，将坐标存放在vbPrevMatched中
// 第一帧  当前帧(第二帧)  第一帧的关键点  待求:对于第一帧关键点索引，对应第二帧中的关键点坐标  100
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0;
    //初始化为-1  对于第一帧中的关键点，在第二帧中有没有对应
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    //?????????????????????????????????
	// HISTO_LENGTH:30
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    //计算旋转的时候要用到
    const float factor = 1.0f/HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
	// 对于第二帧中的特征点，在第一帧中有没有对应
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

	// 遍历第一帧的关键点
    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        // octave是关键点所在的金字塔层数
        int level1 = kp1.octave;
        if(level1>0)
            continue;

        // 根据第一帧关键点xy位置，返回在当前帧中窗口内符合条件(在同一尺度下的)的关键点索引vIndices2找到的都是第二帧中的关键点)
		// 第一帧特征点的xy坐标 windowSize=10  特征点所在的尺度
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

        if(vIndices2.empty())
            continue;

        //参考帧的描述子
        cv::Mat d1 = F1.mDescriptors.row(i1);

        //最好匹配的距离
        int bestDist = INT_MAX;
        //次好匹配的距离
        int bestDist2 = INT_MAX;
        
        //最好匹配的索引
        int bestIdx2 = -1;

		// 遍历当前帧中待匹配的关键点
        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;
            //当前帧的描述子
            cv::Mat d2 = F2.mDescriptors.row(i2);
			//????????????????????怎么计算的  详细看一下
            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

		// 50
        if(bestDist<=TH_LOW)
        {
			// mfNNratio:0.6
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                //如果21中已经与前面的1中有了匹配，则删除该匹配
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                //12中存放的是第一帧关键点最佳匹配的2中的索引
				// 21中存放的是最佳匹配的1中的索引
				//distance存放但是最佳匹配的distance
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                
                //nmatches是匹配的数目
                nmatches++;

				// true
                if(mbCheckOrientation)
                {
                    //angle表示关键点的方向
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
					// factor是1/30
                    int bin = round(rot*factor);
                    
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                    //rotHist是关于匹配点相差角度的直方图
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

	// true
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        //计算直方图中三个最大值对应的索引ind1,ind2,ind3
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            
            //rosHist??????????????????????????????????????
            //除了前三大的直方条，其他的对应都删除
			// 检查旋转  将对应关键点角度最多的留下来了，其他的都删掉
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
				// rotHist[i][j]存放的是第一帧中关键点的索引
                int idx1 = rotHist[i][j];

                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}



// 当前关键帧   候选回环关键帧
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
	// 分别获得当前关键帧和候选回环关键帧的关键点，关键点的特征向量，关联地图点  描述子
    vector<cv::KeyPoint> vKeysUn1 = pKF1->GetKeyPointsUn();
    DBoW2::FeatureVector vFeatVec1 = pKF1->GetFeatureVector();
    vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    cv::Mat Descriptors1 = pKF1->GetDescriptors();

    vector<cv::KeyPoint> vKeysUn2 = pKF2->GetKeyPointsUn();
    DBoW2::FeatureVector vFeatVec2 = pKF2->GetFeatureVector();
    vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    cv::Mat Descriptors2 = pKF2->GetDescriptors();

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    size_t idx1 = f1it->second[i1];

                    MapPoint* pMP1 = vpMapPoints1[idx1];
                    if(!pMP1)
                        continue;
                    if(pMP1->isBad())
                        continue;

                    cv::Mat d1 = Descriptors1.row(idx1);

                    int bestDist1=INT_MAX;
                    int bestIdx2 =-1 ;
                    int bestDist2=INT_MAX;

                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        size_t idx2 = f2it->second[i2];

                        MapPoint* pMP2 = vpMapPoints2[idx2];

                        if(vbMatched2[idx2] || !pMP2)
                            continue;

                        if(pMP2->isBad())
                            continue;

                        cv::Mat d2 = Descriptors2.row(idx2);

                        int dist = DescriptorDistance(d1,d2);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }

                    if(bestDist1<TH_LOW)
                    {
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                        {
							// 对于当前关键帧的关键点，在参考帧中对应点对应的地图点
                            vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2]=true;

                            if(mbCheckOrientation)
                            {
                                float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=NULL;
                //vnMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    return nmatches;
}




// SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedKeysUn1,vMatchedKeysUn2,vMatchedIndices);
// 寻找当前关键帧 与 参考关键帧  没有与地图点相关联的关键点之间的匹配
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
vector<cv::KeyPoint> &vMatchedKeys1, vector<cv::KeyPoint> &vMatchedKeys2, vector<pair<size_t, size_t> > &vMatchedPairs)
{
    //当前帧的地图点
    vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    //当前帧的关键点
    vector<cv::KeyPoint> vKeysUn1 = pKF1->GetKeyPointsUn();
    //当前帧的描述子
    cv::Mat Descriptors1 = pKF1->GetDescriptors();
    //当前帧的特征向量
    DBoW2::FeatureVector vFeatVec1 = pKF1->GetFeatureVector();



    vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    vector<cv::KeyPoint> vKeysUn2 = pKF2->GetKeyPointsUn();
    cv::Mat Descriptors2 = pKF2->GetDescriptors();
    DBoW2::FeatureVector vFeatVec2 = pKF2->GetFeatureVector();
    
    
    

    // Find matches between not tracked keypoints
    // Matching speeded-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(vKeysUn2.size(),false);
    vector<int> vMatches12(vKeysUn1.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::iterator f2end = vFeatVec2.end();

    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                
                size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1];

                // If there is already a MapPoint skip
                if(pMP1)
                    continue;

				// 寻找没有匹配地图点的关键点
                const cv::KeyPoint &kp1 = vKeysUn1[idx1];

                cv::Mat d1 = Descriptors1.row(idx1);

                // 对应某个关键点的匹配关键点 及其距离
                vector<pair<int,size_t> > vDistIndex;


                // 对于没有匹配地图点的特征点kp1以及对应的描述子d1,遍历参考帧所有的关键点
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    cv::Mat d2 = Descriptors2.row(idx2);

                    const int dist = DescriptorDistance(d1,d2);

                    if(dist>TH_LOW)
                        continue;

                    vDistIndex.push_back(make_pair(dist,idx2));
                }

                if(vDistIndex.empty())
                    continue;
                
                
                
                // 得到最好的匹配点
                sort(vDistIndex.begin(),vDistIndex.end());
                int BestDist = vDistIndex.front().first;
                int DistTh = round(2*BestDist);

                for(size_t id=0; id<vDistIndex.size(); id++)
                {
                    //距离不大于最小距离的两倍
                    if(vDistIndex[id].first>DistTh)
                        break;

					// 得到在参考帧中的候选关键点
                    int currentIdx2 = vDistIndex[id].second;
                    cv::KeyPoint &kp2 = vKeysUn2[currentIdx2];
                    
                    // 当前关键帧中的关键点  参考帧中的匹配关键点  基础矩阵 参考关键帧
                    // 如果满足极线约束
                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
                        vbMatched2[currentIdx2]=true;
                        vMatches12[idx1]=currentIdx2;
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            float rot = kp1.angle-kp2.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
						// 如果找到一组匹配，直接返回
                        break;
                    }

                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }



    vMatchedKeys1.clear();
    vMatchedKeys1.reserve(nmatches);
    vMatchedKeys2.clear();
    vMatchedKeys2.reserve(nmatches);
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;

        vMatchedKeys1.push_back(vKeysUn1[i]);
        vMatchedKeys2.push_back(vKeysUn2[vMatches12[i]]);
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}


// th默认为2.5
// 候选关键帧  当前帧的地图点
int ORBmatcher::Fuse(KeyFrame *pKF, vector<MapPoint *> &vpMapPoints, float th)
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    const int nMaxLevel = pKF->GetScaleLevels()-1;
    vector<float> vfScaleFactors = pKF->GetScaleFactors();

    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;


    // 遍历当前帧关联的地图点
    for(size_t i=0; i<vpMapPoints.size(); i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
            continue;

        if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();

        // 地图点在参考帧坐标系下的坐标
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // 重投影到参考帧
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;


        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const float ratio = dist3D/minDistance;

        vector<float>::iterator it = lower_bound(vfScaleFactors.begin(), vfScaleFactors.end(), ratio);
        const int nPredictedLevel = min(static_cast<int>(it-vfScaleFactors.begin()),nMaxLevel);

        // Search in a radius
        const float radius = th*vfScaleFactors[nPredictedLevel];

        vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            const int kpLevel= pKF->GetKeyPointScaleLevel(idx);

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            cv::Mat dKF = pKF->GetDescriptor(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            //如果最优匹配的像素点已经对应一个地图点 将该像素点替换为与最优匹配关联
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    pMP->Replace(pMPinKF);                
            }
            // 否则 将该地图点 与候选关键帧的最优匹配对应
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}



//matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4);
// 邻居关键帧  邻居关键帧矫正后的位姿  所有邻居关键帧关联的地图点
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;



    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    const int nMaxLevel = pKF->GetScaleLevels()-1;
    vector<float> vfScaleFactors = pKF->GetScaleFactors();

    int nFused=0;

    // For each candidate MapPoint project and match
    // 遍历  所有邻居关键帧关联的地图点
    for(size_t iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        // 利用矫正后的位姿 得到的在当前帧坐标系下的三维坐标
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // 重投影
        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const float ratio = dist3D/minDistance;

        vector<float>::iterator it = lower_bound(vfScaleFactors.begin(), vfScaleFactors.end(), ratio);
        const int nPredictedLevel = min(static_cast<int>(it-vfScaleFactors.begin()),nMaxLevel);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF->GetScaleFactor(nPredictedLevel);

        // 搜索范围
        vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int kpLevel = pKF->GetKeyPointScaleLevel(idx);

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            cv::Mat dKF = pKF->GetDescriptor(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            // 如果这个最佳匹配像素点有关联的地图点 则用当前地图点进行替换
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    pMPinKF->Replace(pMP);
            }
            // 否则 进行添加
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }

    }

    return nFused;

}



// SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);
// 当前关键帧 候选回环关键帧  最优匹配的地图点  s R t threshold
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                                   const float &s12, const cv::Mat &R12, const cv::Mat &t12, float th)
{
    const float fx = pKF1->fx;
    const float fy = pKF1->fy;
    const float cx = pKF1->cx;
    const float cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    // 相对  相似变换矩阵
    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;


    // 当前关键帧关联的地图点  数目N1
    const int nMaxLevel1 = pKF1->GetScaleLevels()-1;
    vector<float> vfScaleFactors1 = pKF1->GetScaleFactors();

    vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();


    // 候选关键帧关联的地图点  数目N2
    const int nMaxLevel2 = pKF2->GetScaleLevels()-1;
    vector<float> vfScaleFactors2 = pKF2->GetScaleFactors();

    vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();


    // 当前关键帧 与最优匹配地图点的关联
    vector<bool> vbAlreadyMatched1(N1,false);
    // 候选回环关键帧 与最优匹配地图点的关联
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }


    // 对于sim3没有匹配上的点  根据重投影 搜索得到的1在2中对应的匹配点
    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);


    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;


        // 遍历根据相似矩阵 没有匹配上的地图点

        // 根据相似矩阵 将1中的地图点重投影到2  得到uv
        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        float invz = 1.0/p3Dc2.at<float>(2);
        float x = p3Dc2.at<float>(0)*invz;
        float y = p3Dc2.at<float>(1)*invz;

        float u = fx*x+cx;
        float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        float maxDistance = pMP->GetMaxDistanceInvariance();
        float minDistance = pMP->GetMinDistanceInvariance();
        float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        float ratio = dist3D/minDistance;

        vector<float>::iterator it = lower_bound(vfScaleFactors2.begin(), vfScaleFactors2.end(), ratio);
        const int nPredictedLevel = min(static_cast<int>(it-vfScaleFactors2.begin()),nMaxLevel2);

        // Search in a radius
        float radius = th*vfScaleFactors2[nPredictedLevel];

        // 得到一个候选区域
        vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;

        // 在候选区域中根据描述子 得到最优的匹配像素位置
        for(vector<size_t>::iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            size_t idx = *vit;

            cv::KeyPoint kp = pKF2->GetKeyPointUn(idx);

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            cv::Mat dKF = pKF2->GetDescriptor(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }



        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }




    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        float invz = 1.0/p3Dc1.at<float>(2);
        float x = p3Dc1.at<float>(0)*invz;
        float y = p3Dc1.at<float>(1)*invz;

        float u = fx*x+cx;
        float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        float maxDistance = pMP->GetMaxDistanceInvariance();
        float minDistance = pMP->GetMinDistanceInvariance();
        float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        float ratio = dist3D/minDistance;

        vector<float>::iterator it = lower_bound(vfScaleFactors1.begin(), vfScaleFactors1.end(), ratio);
        const int nPredictedLevel = min(static_cast<int>(it-vfScaleFactors1.begin()),nMaxLevel1);



        // Search in a radius of 2.5*sigma(ScaleLevel)
        float radius = th*vfScaleFactors1[nPredictedLevel];

        vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            size_t idx = *vit;

            cv::KeyPoint kp = pKF1->GetKeyPointUn(idx);

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            cv::Mat dKF = pKF1->GetDescriptor(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }





    // Check agreement
    int nFound = 0;
    //  如果1到2 2到1是同一个点 也将对应扩充到vpMatches12中
    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}





int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, float th)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    //遍历参考帧中的地图点
    for(size_t i=0, iend=LastFrame.mvpMapPoints.size(); i<iend; i++)
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                // Project
                //第一帧中的三维坐标
                cv::Mat x3Dw = pMP->GetWorldPos();
                //当前帧中的三维坐标
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

               int nPredictedOctave = LastFrame.mvKeys[i].octave;

                // Search in a window. Size depends on scale
                // 根据尺度确定半径搜索范围
                float radius = th*CurrentFrame.mvScaleFactors[nPredictedOctave];
                //在重投影的像素上返回候选区域
                vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nPredictedOctave-1, nPredictedOctave+1);

                if(vIndices2.empty())
                    continue;

                //描述子
                cv::Mat dMP = LastFrame.mDescriptors.row(i);

                int bestDist = INT_MAX;
                int bestIdx2 = -1;

                for(vector<size_t>::iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    cv::Mat d = CurrentFrame.mDescriptors.row(i2);

                    int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=TH_HIGH)
                {
                    //设置匹配上的关键点所对应的地图点
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }  

   //Apply rotation consistency
   if(mbCheckOrientation)
   {
       int ind1=-1;
       int ind2=-1;
       int ind3=-1;

       ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

       for(int i=0; i<HISTO_LENGTH; i++)
       {
           if(i!=ind1 && i!=ind2 && i!=ind3)
           {
               for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
               {
                   CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                   nmatches--;
               }
           }
       }
   }

   return nmatches;
}

//int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, float th ,int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;


    //遍历参考帧的地图点
    vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //重投影
                //Project
                //参考帧坐标系下的三维坐标
                cv::Mat x3Dw = pMP->GetWorldPos();
                //当前帧坐标系下的三维坐标
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                float minDistance = pMP->GetMinDistanceInvariance();
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);
                float ratio = dist3D/minDistance;


                vector<float>::iterator it = lower_bound(CurrentFrame.mvScaleFactors.begin(), CurrentFrame.mvScaleFactors.end(), ratio);
                const int nPredictedLevel = min(static_cast<int>(it-CurrentFrame.mvScaleFactors.begin()),CurrentFrame.mnScaleLevels-1);

                // Search in a window
                float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                // 在uv搜索区域中寻找可能的匹配
                vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = INT_MAX;
                int bestIdx2 = -1;

                for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    cv::Mat d = CurrentFrame.mDescriptors.row(i2);

                    int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->GetKeyPointUn(i).angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }


   if(mbCheckOrientation)
   {
       int ind1=-1;
       int ind2=-1;
       int ind3=-1;

       ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

       for(int i=0; i<HISTO_LENGTH; i++)
       {
           if(i!=ind1 && i!=ind2 && i!=ind3)
           {
               for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
               {
                   CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                   nmatches--;
               }
           }
       }
   }

    return nmatches;
}


//ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{

    int max1=0;
    int max2=0;
    int max3=0;


    //得到前三个最长的直方条 和对应的索引
    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }


    //次大和第三大不能小于前一个等级的十分之一
    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
