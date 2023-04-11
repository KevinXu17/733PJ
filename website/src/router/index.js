import { createRouter, createWebHistory } from 'vue-router'
import theHome from '../views/Home.vue'
import theBeauty from '../views/Beauty.vue'

import Recommend from '../views/Recommend.vue'
import SentimentAnalysis from '../views/Sentiment.vue'
import ReviewSummarization from '../views/Summarization.vue'
import PCadv from '../views/PCadv.vue'
import KNNgraph from '../views/KNNgraph.vue'
import SentimentCls from '../views/SentimentCls.vue'
import theClassification from '../views/Classification.vue'


const routes = [
  {
    path: '/',
    name: 'Home',
    component: theHome,
  },
  {
    path: '/beauty',
    name: 'Beauty',
    component: theBeauty,
  },
  {
    path: '/pcadv',
    name: 'PCadv',
    component: PCadv,
  },
  {
    path: '/sentimentcls',
    name: 'SentimentCls',
    component: SentimentCls,
  },
  {
    path: '/knngraph',
    name: 'KNNgraph',
    component: KNNgraph,
  },
  {
    path: '/classification',
    name: 'theClassification',
    component: theClassification,
  },
  {
    path: '/recommend',
    name: 'theRecommend',
    component: Recommend,
  },
  {
    path: '/sentiment',
    name: 'Sentiment',
    component: SentimentAnalysis,
  },
  {
    path: '/summarization',
    name: 'Summarization',
    component: ReviewSummarization,
  },
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
