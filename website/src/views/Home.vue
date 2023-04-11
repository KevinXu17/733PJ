<template>

  <el-button text @click="centerDialogVisible = true" style="margin-top: 20px;">
      Dataset Preview
    </el-button>

    <el-dialog v-model="centerDialogVisible" title="Preview" width="70%" center>
      <img alt="image load failed" src="../img/preview_pc1.png" style="width:100%"/>
    </el-dialog>


    <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <h2>Number of reviews evolved over years</h2>
      </div>
      </template>

      
      <img alt="image load failed" src="../img/kelvin1.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="The number of reviews have evolved over years jump exponentially"
      >
        <template #reference>
          <el-button class="m-2">Graph description</el-button>
        </template>
      </el-popover>
      </div>
    

    </el-card>

    <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <h2>Star Rating Distribution</h2>
      </div>
    </template>

    <img alt="image load failed" src="../img/kelvin2.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="Star rating distribution, more than 40% is 5 star rating, and 25% is 1 star rating."
      >
        <template #reference>
          <el-button class="m-2">Graph description</el-button>
        </template>
      </el-popover>
      </div>
    </el-card>

    <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <h2>Vote Ratio Distribution by Star Rating</h2>
      </div>
    </template>

    <img alt="image load failed" src="../img/kelvin3.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="vote_ratio distribution by star rating. More than half of reviews with 5 star rating or 4 star rating have 90% helpfulness vote ratio."
      >
        <template #reference>
          <el-button class="m-2">Graph description</el-button>
        </template>
      </el-popover>
      </div>
    </el-card>

    <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <h2>Vote Ratio Distribution by Sentence_count</h2>
      </div>
      </template>

      <img alt="image load failed" src="../img/kelvin4.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="vote_ratio distribution by sentence_count. Most of the reviews which has more than 20 sentences have 80% helpfulness vote ratio"
      >
        <template #reference>
          <el-button class="m-2">Graph description</el-button>
        </template>
      </el-popover>
      </div>
    </el-card>


</template>

<script>


export default {
  name: 'theHome',
  props: {
    msg: String
  },
  components: {

  },
  data() {
    return {
      centerDialogVisible: false,

    };
  },
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.box-card{
  margin:5% 10%
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
}
</style>
