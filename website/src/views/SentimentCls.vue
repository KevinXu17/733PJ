<template>
  <el-button text @click="centerDialogVisible = true" style="margin-top: 20px;">
      Dataset Preview
    </el-button>

    <el-dialog v-model="centerDialogVisible" title="Preview" width="40%" center>
      <img alt="image load failed" src="../img/preview_sentcls.png" style="width:80%"/>
    </el-dialog>


  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <h2>Negative VS Positive Reviews</h2>
      </div>
    </template>

    <img alt="image load failed" src="../img/tony3.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="As we can see from the mean and standard deviation shown above, the number of positive and negative reviews is roughly equal, indicating that the data set is of well-defined and there is no obvious class imbalance."
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
  name: 'SentimentCls',
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
