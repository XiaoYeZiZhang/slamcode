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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include <ros/ros.h>

using namespace std;

namespace ORB_SLAM
{

KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}

// 每个词  在哪个关键帧中出现???
void KeyFrameDatabase::add(KeyFrame *pKF)
{
    boost::mutex::scoped_lock lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    boost::mutex::scoped_lock lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

// 参数: 当前关键帧 当前关键帧与具有最多共视点关键帧的最小词袋向量分数
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    // 获得共视图中与当前关键帧相连的关键帧
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    
    
    // 共享了至少一个词袋的关键帧列表  关键帧mnLoopWords存储共同词袋的数目
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        boost::mutex::scoped_lock lock(mMutex);

        // 遍历当前帧的词袋向量
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            // 寻找共同拥有某个词袋的关键帧lKFs
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            // 遍历关键帧lKFs
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                
                if(pKFi->mnLoopQuery!=pKF->mnId)
                {
                    pKFi->mnLoopWords=0;
                    // 将与当前帧相连的局部关键帧剔除，然后遍历与当前关键帧有相同单词的关键帧
                    if(!spConnectedKeyFrames.count(pKFi))
                    {
                        pKFi->mnLoopQuery=pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                // 拥有共同词袋的数目
                pKFi->mnLoopWords++;
            }
        }
    }
    
    
    
    

    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();



    // 候选关键帧 词袋分数
    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    // 最多共享词袋数目
    int maxCommonWords=0;
    
    //遍历共享词袋的关键帧  得到最多共享数目
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }


    int minCommonWords = maxCommonWords*0.8f;

    // 符合条件的关键帧数目
    int nscores=0;


    // 第一次筛选
    // Compute similarity score. Retain the matches whose score is higher than minScore
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        //只有共享词袋数大于最多的80%
        if(pKFi->mnLoopWords>minCommonWords)
        {
            nscores++;
            // 得到候选关键帧与当前关键帧词袋的分数
            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

            pKFi->mLoopScore = si;
            // 分数要比与共视关键帧最小分数(参数)大
            if(si>=minScore)
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();




    // 联合分数  最优关键帧
    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    // 最优联合分数
    float bestAccScore = minScore;


    // 第二次筛选
    // Lets now accumulate score by covisibility
    // 遍历候选关键帧
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        // 得到候选关键帧共视点最多的10个关键帧
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        // 所有关联的候选关键帧分数
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi;
        
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            // 如果候选关键帧的邻居 也是我们得到的候选关键帧 而且共享词袋数大于最大值80&
            // 得到符合条件 而且分数最大的候选关键帧
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore+=pKF2->mLoopScore;
                if(pKF2->mLoopScore>bestScore)
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;


    set<KeyFrame*> spAlreadyAddedKF;
    // 最终得到的候选回环关键帧
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
}



// 检测进行重定位的候选关键帧  匹配有更多共享单词的关键帧  再匹配 有更多共视点的关键帧
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalisationCandidates(Frame *F)
{
    //保存共有单词的关键帧
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        boost::mutex::scoped_lock lock(mMutex);

        //遍历当前帧的所有单词
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            //拥有该单词的关键帧列表
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            //遍历所有拥有该单词的关键帧
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                
                //对于之前还没有涉及到的关键帧  从0开始
                if(pKFi->mnRelocQuery!=F->mnId)
                {
                    pKFi->mnRelocWords=0;
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    
    
    
    
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();



    // Only compare against those keyframes that share enough words
    
    int maxCommonWords=0;
    //得到共享最多的单词数目
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        // mnRelocWords 是某一个关键帧与当前帧共享的单词数
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    // 共享单词数大于最大共享单词数80%的关键帧以及 两个关键帧的相似度
    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        //如果某个关键帧共享的单词数大于最大共享单词数的80% 
        if(pKFi->mnRelocWords>minCommonWords)
        {
            nscores++;
            // si是两个关键帧的相似度???????????????????????????????????????????????
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();



    //10+1总分  最优关键帧
    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    //最优10+1
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    // 遍历共享单词大于80%的关键帧
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        //这些关键帧mvpOrderedConnectedKeyFrames变量为排好序的大于15个共视点的关键帧
        // 选取10个关键帧  共视点最多的10个关键帧
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        //最优分
        float bestScore = it->first;
        //10+1个总分
        float accScore = bestScore;
        //最优关键帧
        KeyFrame* pBestKF = pKFi;
        
        //遍历这10个关键帧  找到这10个关键帧和pKFi中score最大的那一个
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;

            accScore+=pKF2->mRelocScore;
            if(pKF2->mRelocScore > bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        
        
        
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }



    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    
    //候选关键帧
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    
    
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
