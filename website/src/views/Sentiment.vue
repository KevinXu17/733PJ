<template>
  <el-alert title="This function is just a Demo, for full function goto following link" type="warning" show-icon />
  <div style="text-align:left">
    <a href="https://github.com/KevinXu17/733PJ/blob/main/sentiment%2Bsummarization.ipynb">https://github.com/KevinXu17/733PJ/blob/main/sentiment%2Bsummarization.ipynb</a>
  </div>
  
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <h2>Sentiment Analysis</h2>
      </div>
    </template>
    <div class="alert alert-danger" role="alert" v-if="errors.length">
      <p v-for="error in errors" :key="error" style="color:red;">{{error}}</p>
    </div>
   <el-form :model="form" label-width="15%">
    <el-form-item label="Sentence">
      <el-input
        v-model="textarea"
        :rows="6"
        type="textarea"
        placeholder="I recently purchased a watch and I am extremely unsatisfied with its quality and durability. The materials used are of the highest standard, making it very hard to wear and tear. Not only is it aesthetically unpleasing, but it also keeps inaccurate time and has many useless features. I have worn it daily for months now and it still looks and functions like new. I would definitely not recommend this watch to anyone who wants a high-quality, reliable timepiece."
        disabled
      />
    </el-form-item>


    <el-form-item>
      <el-button type="primary" @click="onSubmit">Submit</el-button>
    </el-form-item>

   </el-form>
   <div style="text-align: left;">
   <div style="font-size:30px">result</div>
    <div v-if="show">Terrible</div>
    </div>

  </el-card>
    
    </template>
    
    <script>
    import { reactive } from 'vue'
    
    export default {
      name: 'SentimentAnalysis',
      components: {
    
      },
      data() {
        return {
          show: false,
          form : reactive({
            sentence: '',
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
    