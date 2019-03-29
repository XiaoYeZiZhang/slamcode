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

#include "Tracking.h"
#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include<opencv2/opencv.hpp>

#include"ORBmatcher.h"
#include"FramePublisher.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>
#include<fstream>


using namespace std;

namespace ORB_SLAM
{

//ORB_SLAM::Tracking Tracker(&Vocabulary, &FramePub, &MapPub, &World, strSettingsFile);
Tracking::Tracking(ORBVocabulary* pVoc, FramePublisher *pFramePublisher, MapPublisher *pMapPublisher, Map *pMap, string strSettingPath):
    mState(NO_IMAGES_YET), mpORBVocabulary(pVoc), mpFramePublisher(pFramePublisher), mpMapPublisher(pMapPublisher), mpMap(pMap),
    mnLastRelocFrameId(0), mbPublisherStopped(false), mbReseting(false), mbForceRelocalisation(false), mbMotionModel(false)
{
    // Load camera parameters from settings file
    //获取参数
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
	/*Camera.fx: 609.2855
	Camera.fy: 609.3422 
	Camera.cx: 351.4274
	Camera.cy: 237.7324*/
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);


	// 相机的畸变参数
	/*Camera.k1: -0.3492
	Camera.k2: 0.1363
	Camera.p1: 0.0
	Camera.p2: 0.0*/
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    DistCoef.copyTo(mDistCoef);

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = 18*fps/30;


    cout << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

	/*Camera.RGB: 1*/
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
	// nFeatures: 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
	//.scaleFactor: 1.2  不同金字塔层数之间的尺度
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
	//nLevels: 8  金字塔层数
    int nLevels = fSettings["ORBextractor.nLevels"];
	//fastTh: 20
    int fastTh = fSettings["ORBextractor.fastTh"];    
	//nScoreType: 1 表示fast角点
    int Score = fSettings["ORBextractor.nScoreType"];

    assert(Score==1 || Score==0);

	// 确定每层要提取的特征点数目和每层的尺度
    mpORBextractor = new ORBextractor(nFeatures,fScaleFactor,nLevels,Score,fastTh);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Fast Threshold: " << fastTh << endl;
    if(Score==0)
        cout << "- Score: HARRIS" << endl;
    else
        cout << "- Score: FAST" << endl;


    // ORB extractor for initialization
    // Initialization uses only points from the finest scale level
	// 这个和前面的有什么区别???????????????????

	// mpORBextractor和mpIniORBextractor根据state的不同进行不同的fast角点提取(参数不同)
	// 只是需要提取的特征点数更多一点
    mpIniORBextractor = new ORBextractor(nFeatures*2,1.2,8,Score,fastTh);  

	// 1
    int nMotion = fSettings["UseMotionModel"];
    mbMotionModel = nMotion;

    if(mbMotionModel)
    {
        mVelocity = cv::Mat::eye(4,4,CV_32F);
        cout << endl << "Motion Model: Enabled" << endl << endl;
    }
    else
        cout << endl << "Motion Model: Disabled (not recommended, change settings UseMotionModel: 1)" << endl << endl;


    tf::Transform tfT;
    tfT.setIdentity();
    // Transfor broadcaster (for visualization in rviz)
    mTfBr.sendTransform(tf::StampedTransform(tfT,ros::Time::now(), "/ORB_SLAM/World", "/ORB_SLAM/Camera"));
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetKeyFrameDatabase(KeyFrameDatabase *pKFDB)
{
    mpKeyFrameDB = pKFDB;
}

//调用GrabImage函数
void Tracking::Run()
{
    ros::NodeHandle nodeHandler;
    ros::Subscriber sub = nodeHandler.subscribe("/camera/image_raw", 1, &Tracking::GrabImage, this);

    ros::spin();
}


void Tracking::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{

    cv::Mat im;
    //转为灰度图
    // Copy the ros image message to cv::Mat. Convert to grayscale if it is a color image.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    ROS_ASSERT(cv_ptr->image.channels()==3 || cv_ptr->image.channels()==1);

    if(cv_ptr->image.channels()==3)
    {
        if(mbRGB)
            cvtColor(cv_ptr->image, im, CV_RGB2GRAY);
        else
            cvtColor(cv_ptr->image, im, CV_BGR2GRAY);
    }
    else if(cv_ptr->image.channels()==1)
    {
        cv_ptr->image.copyTo(im);
    }



	// 处理第一帧图像时 mState为NO_IMAGES_YET
    //Frame:提取关键点，去畸变，构建金字塔，将关键点匹配到Grid cells
	/*初始化frame:首先在构建图像金字塔，然后在每一层上提取关键点和描述子，对提取的关键点进行去畸变处理，对于第一帧，对图像的四个角进行去畸变处理
	对图像构建64*48的grid，将提取的特征点按照位置分配到不同的cell中去*/
    if(mState==WORKING || mState==LOST)
        mCurrentFrame = Frame(im,cv_ptr->header.stamp.toSec(),mpORBextractor,mpORBVocabulary,mK,mDistCoef);
    else
        mCurrentFrame = Frame(im,cv_ptr->header.stamp.toSec(),mpIniORBextractor,mpORBVocabulary,mK,mDistCoef);

    // Depending on the state of the Tracker we perform different tasks
    // Tracking默认的参数是NO_IMAGE_YET  就是还没有前一帧的情况
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;
	


    if(mState==NOT_INITIALIZED)
    {
        //mvbPrevMatched存放当前帧去畸变关键点坐标，设置初始化的标准差和迭代次数etc
        //将mState设置为INITIALIZING
        FirstInitialization();
    }

    else if(mState==INITIALIZING)
    {
        Initialize();
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial Camera Pose Estimation from Previous Frame (Motion Model or Coarse) or Relocalisation
        if(mState==WORKING && !RelocalisationRequested())
        {
            if(!mbMotionModel || mpMap->KeyFramesInMap()<4 || mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                // 通过关键点的遍历得到匹配
                //先寻找匹配点  进行BA  找到离群点  对于离群点 通过重投影误差重新搜索最可能的匹配点    将最后找到的所有匹配点 进行BA  得到位姿
                bOK = TrackPreviousFrame();
            else
            {
                // 因为已知当前帧的位姿估计R t  根据R t将地图点进行重投影 搜索得到匹配点
                //先寻找匹配点 进行BA  得到位姿 舍弃离群点
                bOK = TrackWithMotionModel();
                if(!bOK)
                    bOK = TrackPreviousFrame();
            }
        }
        else
        {
            //利用候选候选关键帧进行重定位  PNP求解
            bOK = Relocalisation();
        }




        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(bOK)
            // 扩展了当前帧 相关的关键帧 以及地图点(重投影)  在扩充的地图点上重新进行BA
            bOK = TrackLocalMap();



        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            mpMapPublisher->SetCurrentCameraPose(mCurrentFrame.mTcw);

            //如果满足一定条件，将当前帧作为新的关键帧插入
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();


            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(size_t i=0; i<mCurrentFrame.mvbOutlier.size();i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }

        if(bOK)
            mState = WORKING;
        else
            mState=LOST;

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                Reset();
                return;
            }
        }

        // Update motion model
        if(mbMotionModel)
        {
            if(bOK && !mLastFrame.mTcw.empty())
            {
                // LastTwc是求得上一帧位姿的逆
                cv::Mat LastRwc = mLastFrame.mTcw.rowRange(0,3).colRange(0,3).t();
                cv::Mat Lasttwc = -LastRwc*mLastFrame.mTcw.rowRange(0,3).col(3);
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                LastRwc.copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                Lasttwc.copyTo(LastTwc.rowRange(0,3).col(3));
                //mVelocity设置为当前帧位姿乘以上一帧位姿的逆
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();
        }

        mLastFrame = Frame(mCurrentFrame);
     }       







    // Update drawer
    mpFramePublisher->Update(this);

    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Rwc = mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3).t();
        // 这里取了当前帧位姿的逆
        cv::Mat twc = -Rwc*mCurrentFrame.mTcw.rowRange(0,3).col(3);
        tf::Matrix3x3 M(Rwc.at<float>(0,0),Rwc.at<float>(0,1),Rwc.at<float>(0,2),
                        Rwc.at<float>(1,0),Rwc.at<float>(1,1),Rwc.at<float>(1,2),
                        Rwc.at<float>(2,0),Rwc.at<float>(2,1),Rwc.at<float>(2,2));



        tf::Vector3 V(twc.at<float>(0), twc.at<float>(1), twc.at<float>(2));

        tf::Transform tfTcw(M,V);

        mTfBr.sendTransform(tf::StampedTransform(tfTcw,ros::Time::now(), "ORB_SLAM/World", "ORB_SLAM/Camera"));
    }

}

// 对于第一帧，执行此函数
void Tracking::FirstInitialization()
{
    //We ensure a minimum ORB features to continue, otherwise discard frame
    //提取的关键点个数大于100才进行初始化
    if(mCurrentFrame.mvKeys.size()>100)
    {
		// 将当前帧和前一帧均设置为第一帧
        mInitialFrame = Frame(mCurrentFrame);
        mLastFrame = Frame(mCurrentFrame);

        //mvPrevMatched保存的是当前帧关键点的坐标
        mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
        for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
            mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

        if(mpInitializer)
            delete mpInitializer;

        //Initializer(referenceFrame,sigme(标准差),IterationNum)
		// 初始化 初始化器...听起来真别扭..
        mpInitializer =  new Initializer(mCurrentFrame,1.0,200);
        
        mState = INITIALIZING;
    }
}


// 找到关键点对  利用八点法得到位姿 三角化得到三维坐标
void Tracking::Initialize()
{
    // Check if current frame has enough keypoints, otherwise reset initialization process
    //如果当前帧的关键点个数小于100，重新初始化
    if(mCurrentFrame.mvKeys.size()<=100)
    {
        fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
        mState = NOT_INITIALIZED;
        return;
    }    

    // Find correspondences
    ORBmatcher matcher(0.9,true);
    // 对于mInitialFrame中的每一个关键点，在当前帧对应的一个窗口范围寻找可能与之匹配的关键点，再经过描述子相似度匹配，将最终找到的匹配关键点坐标存放在mvbPrevMatched中,mvIniMatches中存放匹配上的索引
    // 第一帧  当前帧(第二帧)  第一帧的关键点  待求：对于第一帧中的关键点，第二帧中匹配的关键点坐标  100窗口大小
	// 返回成功匹配的关键点个数
	int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

    // Check if there are enough correspondences
    //如果匹配上的点数太少重新初始化
    if(nmatches<100)
    {
        mState = NOT_INITIALIZED;
        return;
    }  

    cv::Mat Rcw; // Current Camera Rotation
    cv::Mat tcw; // Current Camera Translation
    vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)


    //Initialize函数进行初始化   根据八点法计算基础矩阵和单应矩阵  得到初始的Rcw  tcw  利用三角化得到对应参考帧的三维坐标  满足条件的bool vector
	// 当前帧 前一帧和当前帧匹配关键点的索引对应  待求R t 地图点三维坐标  属于内点的地图点
    if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
    {
        for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
        {
            if(mvIniMatches[i]>=0 && !vbTriangulated[i])
            {
                mvIniMatches[i]=-1;
                nmatches--;
            }
        }
        //进行了全局BA  得到关键帧的位姿和地图点的三维位置
        //state设置为WORKING
        CreateInitialMap(Rcw,tcw);
    }

}



void Tracking::CreateInitialMap(cv::Mat &Rcw, cv::Mat &tcw)
{
    // Set Frame Poses
    mInitialFrame.mTcw = cv::Mat::eye(4,4,CV_32F);
    
    //将Rt合并为mTcw
    mCurrentFrame.mTcw = cv::Mat::eye(4,4,CV_32F);
    Rcw.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));
    tcw.copyTo(mCurrentFrame.mTcw.rowRange(0,3).col(3));

    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    //计算词袋 ???????????????????????????
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();


    // Insert KFs in the map
    //将关键帧加入地图Map  
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    //遍历每一关键点对
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        //mvIniP3D为计算得到的第一帧坐标系的关键点的三维坐标
        //得到每个关键点对的三维坐标
        cv::Mat worldPos(mvIniP3D[i]);

        // pKFcur是当前关键帧
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        //每个关键帧有一个mvpMapPoints变量 保存那个关键点对应的mapPoint(相对于参考帧的三维坐标)
        //将mapPoint与关键帧相关联
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);


        //对于地图点mapPoint有一个变量smap<KeyFrame*,size_t> mObservations  保存地图点在哪个关键帧中被看到，对应的像素点索引为idx
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        
        //遍历能够看到该地图点的所有关键帧对应的描述子
		// 得到一个关键帧 它对应的关键点描述子到其他关键帧对应关键点描述子的平均距离最小
        pMP->ComputeDistinctiveDescriptors();

        //获得三维点到能看到它的所有关键帧相机光心的向量平均mNormalVector以及最大最小距离范围mfMinDistance和mfMaxDistance
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        //对于当前帧，保存对应关键点的三维坐标 mapPoint
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;

        //Add to Map
        //将mapPoint加入Map地图  mspMapPoints集合中   mbMapUpdated设置为true
        mpMap->AddMapPoint(pMP);

    }


    
    // Update Connections
    //对于关键帧  对于有共视点的关键帧 建立权重链接  对于那些符合条件的关键帧 更新他们的mConnectedKeyFrameWeights变量
    //对于本关键帧，也更新mConnectedKeyFrameWeights变量  同时建立生成树
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();


    // Bundle Adjustment 全局BA
    ROS_INFO("New Map created with %d points",mpMap->MapPointsInMap());
    //void static GlobalBundleAdjustemnt(Map* pMap, int nIterations=5, bool *pbStopFlag=NULL);
    //全局BA  对于地图Map中的所有关键帧和地图中的所有地图点   优化关键帧的位姿和地图点的三维坐标
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);


    // Set median depth to 1
    //返回的是第一个关键帧对应关键点的三维坐标在当前帧坐标系下的深度值中位数
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;


    //TrackedMapPoints返回当前帧对应的已经确定的三维点数目
    if(medianDepth<0 || pKFcur->TrackedMapPoints()<100)
    {
        ROS_INFO("Wrong initialization, reseting...");
        Reset();
        return;
    }

    //单目相机的初始化得到的深度信息具有不确定性，所以需要将得到的地图点的深度进行归一化，并且将位移向量也进行归一化，
    //将当前帧的位移t 除以平均深度值??????????????????????????????????
    // 这里对位移向量进行归一化
    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);



    // Scale points
    //vpAllMapPoints = mvpMapPoints
    // 这里将地图点的深度进行归一化
    //将匹配上的三维点xyz也乘以平均深度的倒数??????????????????????????????????????????
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }
    
    
    //将关键帧插入 mlNewKeyFrames中
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    //当前帧的位姿就是当前关键帧的位姿
    mCurrentFrame.mTcw = pKFcur->GetPose().clone();
    //设置上一帧 上一个关键帧ID和上一个关键帧
    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;


    //将这两个关键帧加入局部关键帧mvpLocalKeyFrames
    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    //将地图点mspMapPoints加入mvLocalMapPoints
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    //设置参考关键帧
    mpReferenceKF = pKFcur;
    //设置地图的参考地图点为mspMapPoints
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapPublisher->SetCurrentCameraPose(pKFcur->GetPose());

    mState=WORKING;
}

//先寻找匹配点  进行BA  找到离群点  对于离群点 通过重投影误差重新搜索最可能的匹配点    将最后找到的所有匹配点 进行BA  得到位姿
bool Tracking::TrackPreviousFrame()
{
    ORBmatcher matcher(0.9,true);
    vector<MapPoint*> vpMapPointMatches;

    // Search first points at coarse scale levels to get a rough initial estimate
	// 进行特征点匹配的时候  确定一下level的范围
    int minOctave = 0;
    int maxOctave = mCurrentFrame.mvScaleFactors.size()-1;
    if(mpMap->KeyFramesInMap()>5)
        minOctave = maxOctave/2+1;

    //对于参考帧地图点对应的关键点，得到当前帧中对应的关键点匹配
    int nmatches = matcher.WindowSearch(mLastFrame,mCurrentFrame,200,vpMapPointMatches,minOctave);


    // If not enough matches, search again without scale constraint
    //如果匹配的点太少，在没有尺度限制的条件下进行查找
    if(nmatches<10)
    {
        nmatches = matcher.WindowSearch(mLastFrame,mCurrentFrame,100,vpMapPointMatches,0);
        if(nmatches<10)
        {
            vpMapPointMatches=vector<MapPoint*>(mCurrentFrame.mvpMapPoints.size(),static_cast<MapPoint*>(NULL));
            nmatches=0;
        }
    }


    //将参考帧的位姿复制给当前帧
    mLastFrame.mTcw.copyTo(mCurrentFrame.mTcw);
    //得到当前帧  地图点与关键点的对应  (是根据地图点找的关键点   而不是根据参考帧的关键点寻找的)
    mCurrentFrame.mvpMapPoints=vpMapPointMatches;



    // If enough correspondeces, optimize pose and project points from previous frame to search more correspondences
    if(nmatches>=10)
    {
    
        // Optimize pose with correspondences
        //优化当前帧的位姿 对于outlier  将mvbOutlier设置为true
        //这一步的BA主要的目的是寻找离群点  然后用Rt计算重投影误差
        Optimizer::PoseOptimization(&mCurrentFrame);

        for(size_t i =0; i<mCurrentFrame.mvbOutlier.size(); i++)
            //将优化过程中发现的outlier对应的mapPoint删除
            if(mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
                mCurrentFrame.mvbOutlier[i]=false;
                nmatches--;
            }

        // Search by projection with the estimated pose
        // 遍历所有的地图点,通过重投影误差  对于离群点的mapPoint,寻找在当前帧中的最佳匹配
        nmatches += matcher.SearchByProjection(mLastFrame,mCurrentFrame,15,vpMapPointMatches);
    }
    else //Last opportunity
        nmatches = matcher.SearchByProjection(mLastFrame,mCurrentFrame,50,vpMapPointMatches);


    mCurrentFrame.mvpMapPoints=vpMapPointMatches;

    if(nmatches<10)
        return false;




    // Optimize pose again with all correspondences
    //根据 通过重投影误差得到的最优匹配点 重新进行BA位姿的估计
    Optimizer::PoseOptimization(&mCurrentFrame);

    //除去离群点  mvpMapPoints中的地图点比参考帧中的mvpMapPoints少
    // Discard outliers
    for(size_t i =0; i<mCurrentFrame.mvbOutlier.size(); i++)
        if(mCurrentFrame.mvbOutlier[i])
        {
            mCurrentFrame.mvpMapPoints[i]=NULL;
            mCurrentFrame.mvbOutlier[i]=false;
            nmatches--;
        }

    return nmatches>=10;
}






bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);
    vector<MapPoint*> vpMapPointMatches;

    // Compute current pose by motion model
    // 根据速度模型得到当前帧的预测  根据上一帧的位姿和位姿的变化得到该帧的位姿估计
    // 有了1 2 当前为3
    //mVelocity为T2*T1-1  T3 = T2*T1-1*T2   T1-1*T2可以理解为从1到2的位姿变换   也就近似与2到3的位姿变换 用2的位姿乘以2-3的位姿变换 就得到了3的位姿预测值
    mCurrentFrame.mTcw = mVelocity*mLastFrame.mTcw;

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    // 因为已知当前帧的位姿估计R t  根据R t将地图点进行重投影 搜索得到匹配点
    // 根据尺度信息 确定半径搜索范围，在该范围中 寻找最佳匹配  设置最佳匹配的地图点
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,15);

    if(nmatches<20)
       return false;

    // Optimize pose with all correspondences
    // 根据当前帧对应的所有地图点  估计当前帧的位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    // 舍弃离群点
    for(size_t i =0; i<mCurrentFrame.mvpMapPoints.size(); i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
                mCurrentFrame.mvbOutlier[i]=false;
                nmatches--;
            }
        }
    }

    return nmatches>=10;
}

bool Tracking::TrackLocalMap()
{
    // Tracking from previous frame or relocalisation was succesfull and we have an estimation
    // of the camera pose and some map points tracked in the frame.
    // Update Local Map and Track

    // Update Local Map
    // 更新局部关键帧  和局部地图点
    UpdateReference();



    // Search Local MapPoints
    // 因为地图点扩充了   对于当前帧，也扩充一些其对应的三维点，将三维点与像素坐标相对应 只遍历当前帧有关联的地图点  在uv区域内搜索与三维坐标对应的最佳像素坐标
    SearchReferencePointsInFrustum();


    // 重新进行位姿优化
    // Optimize Pose
    // 进行位姿估计(扩充后的)
    mnMatchesInliers = Optimizer::PoseOptimization(&mCurrentFrame);



    // Update MapPoints Statistics
    for(size_t i=0; i<mCurrentFrame.mvpMapPoints.size(); i++)
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
				// mapPoint的mnFound++
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
        }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}



//返回当前帧是否作为关键帧插入
bool Tracking::NeedNewKeyFrame()
{
    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    // Not insert keyframes if not enough frames from last relocalisation have passed
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mpMap->KeyFramesInMap()>mMaxFrames)
        return false;

    // Reference KeyFrame MapPoints
    // 与当前帧共视点最多的关键帧相关的地图点数目
    int nRefMatches = mpReferenceKF->TrackedMapPoints();

    // Local Mapping accept keyframes?  bool值
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();



    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle空转
    const bool c1b = mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle;
    // Condition 2: Less than 90% of points than reference keyframe and enough inliers
    //mnMatchesInliers为当前帧BA后  非离群点的数目
    //nRefMatches为参考关键帧相关的地图点数目
    const bool c2 = mnMatchesInliers<nRefMatches*0.9 && mnMatchesInliers>15;


    
    if((c1a||c1b)&&c2)
    {
        // If the mapping accepts keyframes insert, otherwise send a signal to interrupt BA, but not insert yet
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            //mbAbortBA设置为True
            mpLocalMapper->InterruptBA();
            return false;
        }
    }
    else
        return false;
}


//创建新的关键帧
void Tracking::CreateNewKeyFrame()
{
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpLocalMapper->InsertKeyFrame(pKF);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}




void Tracking::SearchReferencePointsInFrustum()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = NULL;
            }
            else
            {
                //mnVisible++
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }


    //更新R t OW
    mCurrentFrame.UpdatePoseMatrices();

    int nToMatch=0;



    // Project points in frame and check its visibility
    // 遍历所有的局部地图点
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        // 如果该地图点已经在当前帧中有对应，直接跳过
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;    
                
        // Project (this fills MapPoint variables for matching)
        //对于地图点  深度Z需要大于0  到相机光心的向量与 地图点的平均距离向量的夹角要满足条件  如果满足条件  那么该地图点mbTrackInView为true
        //设置像素坐标 uv level和夹角
        // 因为地图点扩充了   对于当前帧，也扩充一些其对应的三维点，将三维点与像素坐标相对应
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }    


    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
            
        // 将局部地图点投影到当前帧 在uv区域内搜索与三维坐标对应的最佳像素坐标
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}




void Tracking::UpdateReference()
{    
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    //更新局部地图  将有共视点的关键帧及一部分邻居加入局部关键帧   同时保存与当前帧共视点最多的关键帧到mpReferenceKF变量
    UpdateReferenceKeyFrames();
    // 将localKeyFrame中的地图点加入局部地图  扩充局部地图点的数量
    UpdateReferencePoints();
}




void Tracking::UpdateReferencePoints()
{
    mvpLocalMapPoints.clear();

    // 遍历 参与局部地图的局部关键帧UpdateReferenceKeyFrames()中得到的
    for(vector<KeyFrame*>::iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        //得到关键帧匹配的地图点
        vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                //将所有localKeyFrame中的地图点加入局部地图  地图点设置mnTrackReferenceForFrame
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


//更新参考关键帧
void Tracking::UpdateReferenceKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    //对于当前帧所有的地图点，关键帧可见的次数
    map<KeyFrame*,int> keyframeCounter;
    
    //遍历当前帧的地图点
    for(size_t i=0, iend=mCurrentFrame.mvpMapPoints.size(); i<iend;i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
             
                map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    int max=0;
    KeyFrame* pKFmax=NULL;


    //将有共视点的关键帧均加入mvpLocalKeyFrames
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    //遍历所有能够看到地图点的关键帧
    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;
        //得到共视点最多的共视点个数和关键帧
        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        // mnTrackReferenceForFrame是啥????????????????
        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    //添加一些邻居
    for(vector<KeyFrame*>::iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        //选取与某个关键帧有最多共视点的10个关键帧  把他们也加入localMap中
        vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
		 
        for(vector<KeyFrame*>::iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

    }

    //mpReferenceKF保存有最多共视点的关键帧
    mpReferenceKF = pKFmax;
}






// 重定位
bool Tracking::Relocalisation()
{
    // Compute Bag of Words Vector
    // 计算当前帧的词袋
    mCurrentFrame.ComputeBoW();

    // Relocalisation is performed when tracking is lost and forced at some stages during loop closing
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs;
    
    
    
    // 候选关键帧 vpCandidateKFs
    if(!RelocalisationRequested())
        // 获得候选关键帧  在所有的关键帧中寻找
        vpCandidateKFs= mpKeyFrameDB->DetectRelocalisationCandidates(&mCurrentFrame);
    else // Forced Relocalisation: Relocate against local window around last keyframe
    {
        boost::mutex::scoped_lock lock(mMutexForceRelocalisation);
        mbForceRelocalisation = false;
        vpCandidateKFs.reserve(10);
        //获得最后一个关键帧9个大于15个共视点的关键帧  将这些关键帧作为候选关键帧
        vpCandidateKFs = mpLastKeyFrame->GetBestCovisibilityKeyFrames(9);
        vpCandidateKFs.push_back(mpLastKeyFrame);
    }



    if(vpCandidateKFs.empty())
        return false;


    //候选关键帧的个数
    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    //对于n个满足条件的候选关键帧  得到n个需要求解的pnpSolver
    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    //候选关键帧是否要丢弃
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    // 满足一定条件的候选关键帧
    int nCandidates=0;


    // 遍历所有的候选关键帧
    for(size_t i=0; i<vpCandidateKFs.size(); i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            // 这一个函数  就是通过词袋匹配   将当前帧 与候选关键帧pKF  进行匹配  如果满足匹配  则将当前帧与候选关键帧的mapPoint相关联  保存在vvpMapPointMatches中
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // ref https://blog.csdn.net/qq_30356613/article/details/80588134
                //如果候选关键帧 与 当前帧有足够多的匹配  进行pnp   3d-2d  已知世界坐标系下的坐标，和对应于该帧中匹配的点的像素坐标，求解相机此时的位姿
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                //设置参数
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }        
    }





    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
    
        // 遍历所有的候选关键帧  估计Rt 有足够多的非离群点 就返回
        for(size_t i=0; i<vpCandidateKFs.size(); i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            //cv::Mat PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }
           
            // 通过PnP计算出当前帧的位姿 进行优化
            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;
                
                
                //vbInliers为PnP求解完后 得到的非离群点
                for(size_t j=0; j<vbInliers.size(); j++)
                {
                    if(vbInliers[j])
                    {
                        // 将pnp求解后的非离群点与地图点的匹配存如mvpMapPoints
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }


                //对pnp求解后的非离群点 进行位姿优化
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(size_t io =0, ioend=mCurrentFrame.mvbOutlier.size(); io<ioend; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=NULL;





                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
					// 对于重投影关键帧的那些没有在误差范围内的地图点（也就是当前帧没有特征点对应）  通过重投影+描述子匹配找到匹配点
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(size_t ip =0, ipend=mCurrentFrame.mvpMapPoints.size(); ip<ipend; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            // 在当前帧中寻找更多与地图点的匹配
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                //位姿优化
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                //舍弃离群点
                                for(size_t io =0; io<mCurrentFrame.mvbOutlier.size(); io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {                    
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::ForceRelocalisation()
{
    boost::mutex::scoped_lock lock(mMutexForceRelocalisation);
    mbForceRelocalisation = true;
    mnLastRelocFrameId = mCurrentFrame.mnId;
}

bool Tracking::RelocalisationRequested()
{
    boost::mutex::scoped_lock lock(mMutexForceRelocalisation);
    return mbForceRelocalisation;
}


void Tracking::Reset()
{
    {
        boost::mutex::scoped_lock lock(mMutexReset);
        mbPublisherStopped = false;
        mbReseting = true;
    }

    // Wait until publishers are stopped
    ros::Rate r(500);
    while(1)
    {
        {
            boost::mutex::scoped_lock lock(mMutexReset);
            if(mbPublisherStopped)
                break;
        }
        r.sleep();
    }

    // Reset Local Mapping
    mpLocalMapper->RequestReset();
    // Reset Loop Closing
    mpLoopClosing->RequestReset();
    // Clear BoW Database
    mpKeyFrameDB->clear();
    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NOT_INITIALIZED;

    {
        boost::mutex::scoped_lock lock(mMutexReset);
        mbReseting = false;
    }
}

void Tracking::CheckResetByPublishers()
{
    bool bReseting = false;

    {
        boost::mutex::scoped_lock lock(mMutexReset);
        bReseting = mbReseting;
    }

    if(bReseting)
    {
        boost::mutex::scoped_lock lock(mMutexReset);
        mbPublisherStopped = true;
    }

    // Hold until reset is finished
    ros::Rate r(500);
    while(1)
    {
        {
            boost::mutex::scoped_lock lock(mMutexReset);
            if(!mbReseting)
            {
                mbPublisherStopped=false;
                break;
            }
        }
        r.sleep();
    }
}

} //namespace ORB_SLAM
