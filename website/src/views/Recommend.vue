<template>
  <el-alert title="This function is just a Demo, for full function goto following link" type="warning" show-icon />
  <div style="text-align:left">
    <a href="https://github.com/KevinXu17/733PJ/blob/main/Amazon_Product_RS.ipynb">https://github.com/KevinXu17/733PJ/blob/main/Amazon_Product_RS.ipynb</a>
  </div>
  
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <h2>Provide products for Comparison given the seller id</h2>
      </div>
    </template>
    <div class="alert alert-danger" role="alert" v-if="errors.length">
      <p v-for="error in errors" :key="error" style="color:red;">{{error}}</p>
    </div>
   <el-form :model="form" label-width="15%">
    <el-form-item label="Seller id">
      <el-input v-model="form.user_id" placeholder="A39HTATAQ9V7YF" disabled/>
    </el-form-item>

    <el-form-item label="Numbers of Comparision">
      <el-input v-model="form.nums" placeholder="5" disabled/>
    </el-form-item>

    <el-form-item>
      <el-button type="primary" @click="onSubmit">Recommend</el-button>
    </el-form-item>

   </el-form>
   <div style="text-align: left;">
   <div style="font-size:30px">result</div>
    <div v-if="show">
      The top 5 recommended products are: B00004TUBL, B004LJ0ZK6, B004HFBC9W, B004HSO124, B004INQ65S (products are provided in ASIN)<br/><br/><br/>
      You can look up ASIN with this link<a href="https://amazon-asin.com/">https://amazon-asin.com/</a><br/>
      The dataset is from 2003, some products are already removed
    </div>
  </div>
  </el-card>
    
    </template>
    
    <script>
    import { reactive } from 'vue'
    
    export default {
      name: 'theRecommend',
      components: {
    
      },
      data() {
        return {
          show: false,
          form : reactive({
            user_id: '',
            nums: '',
          }),
          errors: [],
          result: '',

        };
      },
      methods: {
        onSubmit(){
          this.show = true
        }
      },
    }
    </script>
    

    <!-- Add "scoped" attribute to limit CSS to this component only -->
    <style scoped>
    .box-card{
      margin:3%;
    }
    h3 {
      margin: 40px 0 0;
    }
    ul {
      list-style-type: none;
      padding: 0;
    }
    li {
      display: inline-block;
      margin: 0 10px;
    }
    a {
      color: #42b983;
      text-align: left;
      width:100%;
    }
    p{
      text-align: left;
    }
    h2{
      text-align: left;
    }
    </style>
    