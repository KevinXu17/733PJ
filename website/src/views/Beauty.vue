<template>
  <el-button text @click="centerDialogVisible = true" style="margin-top: 20px;">
      Dataset Preview
    </el-button>

    <el-dialog v-model="centerDialogVisible" title="Preview" width="40%" center>
      <img alt="image load failed" src="../img/preview_beauty.png" style="width:80%"/>
    </el-dialog>

  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <h2>Beauty Product Rating Count</h2>
      </div>
    </template>

    <img alt="image load failed" src="../img/tony1.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="the majority of ratings are 5-star ratings, followed by 4-star ratings . The remaining ratings are fairly evenly distributed, with 3-star ratings being slightly more common than 1-star or 2-star ratings"
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
        <h2>Top 5 most popular beauty product</h2>
      </div>
    </template>

    <img alt="image load failed" src="../img/tony2.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="The top 5 most popular beauty product, product is given in ASIN"
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
  name: 'theBeauty',
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
