# 733PJ

### 1 Data Cleaning
#### 1.1. Data Link
Since the rawData and the data after cleaning are large, so we upload the data in Google Drive. Please download the datasets and put them to the right path.

Original Data Link: https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt
Google Drive Link: https://drive.google.com/file/d/14irqyE21ooYZhzX4A-AAAScZECqnOmX3/view?usp=sharing

#### 1.2. File Path  

#### 1.3 Format
Raw Data Columns:

marketplace       - 2 letter country code of the marketplace where the review was written. <br>
customer_id       - Random identifier that can be used to aggregate reviews written by a single author. <br>
review_id         - The unique ID of the review. <br>
product_id        - The unique Product ID the review pertains to. In the multilingual dataset the reviews
                    for the same product in different countries can be grouped by the same product_id. <br>
product_parent    - Random identifier that can be used to aggregate reviews for the same product. <br>
product_title     - Title of the product. <br>
product_category  - Broad product category that can be used to group reviews 
                    (also used to group the dataset into coherent parts). <br>
star_rating       - The 1-5 star rating of the review. <br>
helpful_votes     - Number of helpful votes. <br>
total_votes       - Number of total votes the review received. <br>
vine              - Review was written as part of the Vine program. <br>
verified_purchase - The review is on a verified purchase. <br>
review_headline   - The title of the review. <br>
review_body       - The review text. <br>
review_date       - The date the review was written. <br>

#### 1.4 Run Data Process
