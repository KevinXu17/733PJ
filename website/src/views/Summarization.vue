<template>
  <el-alert title="This function is just a Demo, for full function goto following link" type="warning" show-icon />
  <div style="text-align:left">
    <a href="https://github.com/KevinXu17/733PJ/blob/main/sentiment%2Bsummarization.ipynb">https://github.com/KevinXu17/733PJ/blob/main/sentiment%2Bsummarization.ipynb</a>
  </div>
  
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <h2>Review Summarization</h2>
      </div>
    </template>
    <div class="alert alert-danger" role="alert" v-if="errors.length">
      <p v-for="error in errors" :key="error" style="color:red;">{{error}}</p>
    </div>
   <el-form :model="form" label-width="15%">
    <el-form-item label="Review">
      <el-input
        v-model="textarea"
        :rows="10"
        type="textarea"
        placeholder="I recently purchased a pair of wireless earbuds and I have to say, they are absolutely fantastic! The sound quality is amazing, 
 with crystal clear audio and deep, rich bass. I was blown away by how immersive my music sounded when I listened to it on these earbuds.
But that's not all - the design is sleek and modern, and the earbuds fit perfectly in my ears without ever feeling uncomfortable or falling out. 
And the charging case is compact and easy to use, making it convenient to keep my earbuds charged up and ready to go.
One of the things I appreciate most about these earbuds is the touch controls - I can easily pause, play, skip tracks, 
and adjust the volume without ever having to take my phone out of my pocket. It's so convenient when I'm on the go or trying to multitask.
Overall, I would highly recommend these wireless earbuds to anyone looking for a high-quality audio experience in a compact, user-friendly package. 
They have truly exceeded my expectations and I can't imagine going back to wired earbuds ever again."
        disabled
      />
    </el-form-item>


    <el-form-item>
      <el-button type="primary" @click="onSubmit">Submit</el-button>
    </el-form-item>

   </el-form>
   <div style="text-align: left;">
   <div style="font-size:30px">result</div>
    <div v-if="show">I recently purchased a pair of wireless earbuds and I have to say, they are absolutely fantastic! But that's not all - the design is sleek and modern, and the earbuds fit perfectly in my ears without ever feeling uncomfortable or falling out.</div>
    </div>

  </el-card>
    
    </template>
    
    <script>
    import { reactive } from 'vue'
    
    export default {
      name: 'ReviewSummarization',
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
    