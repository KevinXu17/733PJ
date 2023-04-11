<template>
  <div class="affix-container">
    <el-affix target=".affix-container">
      <el-menu
        :default-active="activeIndex"
        class="el-menu-demo"
        mode="horizontal"
        background-color="#545c64"
        text-color="#fff"
        active-text-color="#ffd04b"
        @select="handleSelect"
      >
      <p style="padding:0px 10px;color:aliceblue;">Amazon Seller CRM</p>
        <el-sub-menu index="1">
          <template #title>General Analysis</template>
          <el-menu-item index="1-1"><router-link to='/' style="color: aliceblue;">PC Product</router-link></el-menu-item>
          <el-menu-item index="1-2"><router-link to='/beauty' style="color: aliceblue;">Beauty Product</router-link></el-menu-item>
        </el-sub-menu>
        <el-sub-menu index="2">
          <template #title>Advanced Analysis</template>
          <el-menu-item index="2-1"><router-link to='/pcadv' style="color: aliceblue;">PC Text Visualization</router-link></el-menu-item>
          <el-menu-item index="2-2"><router-link to='/knngraph' style="color: aliceblue;">KNN Graph</router-link></el-menu-item>
          <el-menu-item index="2-3"><router-link to='/sentimentcls' style="color: aliceblue;">Sentiment Classification</router-link></el-menu-item>
          <el-menu-item index="2-4"><router-link to='/classification' style="color: aliceblue;">RFT and MLP classification</router-link></el-menu-item>
        </el-sub-menu>
        <el-sub-menu index="3">
          <template #title>More Features</template>
          <el-menu-item index="3-1"><router-link to='/recommend' style="color: aliceblue;">Product Comparison</router-link></el-menu-item>
          <el-menu-item index="3-2"><router-link to='/sentiment' style="color: aliceblue;">Sentiment Analysis</router-link></el-menu-item>
          <el-menu-item index="3-3"><router-link to='/summarization' style="color: aliceblue;">Review Summarization</router-link></el-menu-item>
        </el-sub-menu>
      </el-menu>
    </el-affix>
    <router-view></router-view>
  <!-- <Home msg="Welcome to Your Vue.js App"/> -->
  </div>
</template>

<script>
// import theHome from './views/Home.vue'
import { ref } from 'vue'

export default {
  name: 'App',
  data() {
    return {
      activeIndex : ref('1-1'),

    };
  },
  components: {
    // theHome
  },
  methods: {
    handleSelect(key, keyPath) {
      // if(key==1){
      //   this.$router.push('/')
      // }
      // if(key==2){
      //   this.$router.push('/review_analysis')
      // }
      console.log(key, keyPath);
    },
  },
}
</script>

<style>
.affix-container{
 margin:0; 
 padding:0
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

body { margin: 0; padding:0}
</style>
