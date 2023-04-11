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
        <h2>Confusion Matrix of RFT Classificaftion - Default Parameter Base Model</h2>
      </div>
    </template>

    <img alt="image load failed" src="../img/kelvin5.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="RFT accuracy is 82.3%"
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
        <h2>Confusion Matrix of MLP Classificaftion - After hyperparameters tuning</h2>
      </div>
    </template>

    <img alt="image load failed" src="../img/kelvin6.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="MLP accuracy after hyperparameters tuning is 85.5%"
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
  name: 'theClassification',
  data() {
    return {
      centerDialogVisible: false,

    };
  },
  props: {
    msg: String
  },
  components: {
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
