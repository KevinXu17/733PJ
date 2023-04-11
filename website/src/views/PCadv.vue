<template>
  <el-button text @click="centerDialogVisible = true" style="margin-top: 20px;">
      Dataset Preview
    </el-button>

    <el-dialog v-model="centerDialogVisible" title="Preview" width="70%" center>
      <img alt="image load failed" src="../img/preview_pc.jpg" style="width:100%"/>
    </el-dialog>

    <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <h2>Distribution of Polarity Score</h2>
      </div>
    </template>

    <img alt="image load failed" src="../img/bai1.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="Polarity scores which range from -1 to 1, quantify sentiments in customer reviews. Polarity score equal to 1 means customers hold positive thoughts. When the polarity score of a review is equivalent to -1, it shows negative emotions. This plot shows the polarity score distribution of more than 30000 reviews on Amazon PC products. The average polarity score in this plot is about 0.145, and the shape of this curve looks like a normal distribution. From this plot, most of the reviews on PC products show neutral emotions. "
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
        <h2>Polarity Score Distribution by Ratings</h2>
      </div>
    </template>

    <img alt="image load failed" src="../img/bai2.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="his plot shows the distribution of polarity scores within different ratings. Customers with a high rating on their purchased products tend to leave more positive reviews. "
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
        <h2>Top 20 most frequent unigrams in reviews</h2>
      </div>
    </template>

    <img alt="image load failed" src="../img/bai3.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="N-grams mean n consecutive words in a sentence. This plot shows the 20 most common unigrams among 30,000 reviews. Except for verbs, most unigrams relate to PC products, such as “screen,” “keyboard,” and “laptop.” The plot gives the business owner a rough idea about their product review."
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
        <h2>Top 20 most frequent bigrams in reviews</h2>
      </div>
    </template>

    <img alt="image load failed" src="../img/bai4.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="This plot shows the most common bigrams among 30,000 reviews. The top one is the “hard drive,” which occurs about 7,000 times. From this plot, we have enough reason to believe customers care about the “hard drive” and “battery life” when shopping for PC products."
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
        <h2>Top 20 most frequent trigrams in reviews</h2>
      </div>
    </template>

    <img alt="image load failed" src="../img/bai5.png" style="width:80%"/>

      <div>
      <el-popover
        placement="bottom"
        :width="600"
        trigger="click"
        content="This plot shows the most common trigrams among 30,000 reviews."
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
  name: 'PCadv',
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
