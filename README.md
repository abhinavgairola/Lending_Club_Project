 <!DOCTYPE html>
<html>
 <body style="background-color: #3c3d41;">
 <h1>Credit default prediction using feature engineering and interpretable ML model on lending club data</h1>
 <!--# Project Name-->
 <!--### Credit default prediction using lending club data.-->
 <p align="center">
 <img src="https://github.com/abhinavgairola/Lending_Club_Project/blob/master/images/Word_Cloud_5.png" width="850" height="850" title="Word cloud image of lending club loan descriptors"><br>
   <!--<p align="center">
 <h2> Word cloud image of lending club loan descriptors</h2><br>
 </p>-->
 </p>
 <h2> This project is a part of the <a href= "https://www.springboard.com/">Springboard's </a> 6 months long immersive program in data science. The project is about classifiying the credit default using the <a href= "https://www.lendingclub.com/"> Lending club </a> data and to understand which factors contributes towards it the most.</h2> 
 <h2> Project status &#x2705;</h2>
 <h2>Project Intro/Objective</h2>
<p>Lending Club is a money borrowing service that is in the business of providing loans for both personal and business purposes. The data they have collected contains 150 features and more than 2 million elements per feature. This is dense data and provides a unique opportunity of understanding if a lender is going to do default in the future or not? If one can come up with a simple model of classifying a future defaulter then it can greatly hedge the risk of the lender. This hedging of the risk with a simple and explanable model is the principal aim of this project. </p>
<h2>The following technologies were used while working on this project</h2>
 <p><ol>
 <li>Coded extensively in <a href="https://www.python.org"> Python </a></li>
 <li> on <a href="https://jupyter.org"> Jupyter</a></li>
 <li> and in <a href="https://colab.research.google.com">Google colab</a></li>
 <li> Visualization done mostly on <a href="https://plotly.com"> plotly </a></li>
 <li> with some plots on <a href="https://matplotlib.org">Matplotlib</a> and <a href="https://seaborn.pydata.org">seaborn</a> </li>
   <li> Machine learning on <a href="https://pycaret.org">Pycaret</a> and <a href="https://scikit-learn.org/stable/"> scikit-learn</a></li>
</ol>

</p>
<h2>The files and folders which contain the work</h2>
<ol>
 <li>
<p> EDA with some parts of data cleaning in a separate file. Since the visualizations are mostly done on <a href="https://plotly.com">plotly</a> and it is difficult to render them correctly on github-- so this link will open it as an html page. 
 <a href="https://abhinavgairola.github.io/Lending_Club_Project/"> Link to EDA notebook (rendered as html webpage)</a></p></li>
 <li> <a href="https://github.com/abhinavgairola/Lending_Club_Project/blob/master/Machine_Learning.ipynb">Data cleaning and machine learning</a></li>
 <li> <a href="https://github.com/abhinavgairola/Lending_Club_Project/blob/master/CatBoost.ipynb">Data cleaning and machine learning</a></li>
  
 </ol>
 <p><a href="https://christophm.github.io/interpretable-ml-book/shap.html"> The ML models were interpreted using the SHapley Additive exPlanations(SHAP) values.</a> More can be learned about them by going to the mentioned link.</p>
 <h1>References</h1>
 <ol>
 <li><cite><a href="https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9">The Search for Categorical Correlation</a></cite> by Shaked Zychlinski
  .</li>
 <li><cite><a href="https://maxhalford.github.io/blog/target-encoding/">Target encoding done the right way</a></cite> by Max Halford
  .</li>
 <li><cite><a href="https://christophm.github.io/interpretable-ml-book/">Interpretable Machine Learning:-A Guide for Making Black Box Models Explainable.</a></cite> by Christoph Molnar
  .</li>
 </ol>



</body>
</html>


<h1> 
<center><b> Classifying credit defaults using interpretable machine learning 
</b></center>
<center><b> A. Gairola 
</b></center>
<center><b> 
</b></center>
<center>
<img src="https://drive.google.com/uc?id=1ERXczWs6_uioWX9imRDA1oSalkDVswVO "alt="Loan Cartoon" width="400" height="300">
</center>
</h1>

<h2>
<section>
<b>Problem Definition</b>
<a name="ref_sec_1">
<p>
Often people tend to lend extra cash from the financial institutions of their choice to pay for home improvements, finance a wedding or consolidate high-interest debt etc. If used wisely (and there is a BIG If), a loan can fill the loan in the lenders budget without risking his/ her home or other assets. To seek a loan the lenders tend to go to a financial institution/ company which offers curated loans suitable for the lenders need. Lending Club is one such company which offers money borrowing service that is the company is in the business of providing loans for both personal and business purposes. They have a collection of the lender's data spanned over a certain period of time. The data which they have collected contains 150 features and more than 2 million elements per feature. This is dense data and provides a unique opportunity of understanding if a lender is going to do default in the future or not? If one can come up with a simple model of classifying a future defaulter then it can greatly hedge the risk of the lender. This hedging of the risk with a simple and explainable model is the principal aim of this project.
</p>
</a>
</section>
</h2>

<h2>
<section>
<b>The Analysis methodology</b>
<p>
For this I used the typical data science methodology <a href="https://aiden-dataminer.medium.com/the-data-science-method-dsm-a-framework-on-how-to-take-your-data-science-projects-to-the-next-91f9fd81e5d1">which comprise of the following 6 steps</a>:
<ol>
<li> <a href="#ref_sec_1">Problem identification</a></li>
<li> <a href="#ref_sec_2">Data Wrangling </a></li>
<li><a href="#ref_sec_3">Exploratory Data Analysis</a> </li>
<li> <a href="#ref_sec_4">Pre-processing and Training Data Development</a></li>
<li><a href="#ref_sec_5">Modeling </a> </li>
<li> <a href="#ref_sec_6">Conclusion </a></li>
</ol>
</p>
</section>
</h2>


<h2>
<section>
<b>Data Wrangling</b>
<a name="ref_sec_2">
<p>
The raw dataset provided by the Lending club contains a lot of features--151 with 2 million elements per features. The primary datatypes available in the this datasets are either object or float. A significant numer of features has a lot of <i>Nan</i>. While working on this dataset I decided to weed out those features which are more than $90\%$ <i>NaNs</i>.
I removed the feature "desc" containining the descriptive statements given by the lenders  <i><font color="#FF0000">("We knew that using our credit cards to finance an adoption would squeeze us, but then medical and other unexpected expenses made the situation almost impossible. We are a stable family in a stable community. We just need to break a cycle of debt that is getting worse.")</font></i> which may be affective if I do some sentiment analysis--however this pipeline doesn't look into that.
</p>
</a>
<a name ="url">
<table border="1" class="dataframe">
<caption><b>URL feature of Lending club</b></caption>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://lendingclub.com/browse/loanDetail.action?loan_id=68407277</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://lendingclub.com/browse/loanDetail.action?loan_id=68355089</td>
    </tr>
  </tbody>
</table>
</a>
<p>
url is a feature where lending club has provided certain links <a href="#url">(as described in the 'url' table)</a> which is clearly not important for classifying a defaulter. Other than this features like "employment title" and "zip codes" were also removed during the wrangling process.</p>
<h3>
<!--<b>Feature leakage</b>
</h3><p> Feature leakage is a situation in which a model is built using data which is not available at the time the model will be used to make a prediction. Considering this and after checking the description of the data in the lending club data dictionary--I will drop these features as these features may not be available at the time of classification and or may be too indicative of the target value.
For example the recoveries feature is the "post charge off recovery"---this is only available after a charge off has occured and shouldn't be included in the model development.</p>-->
<figure>
<center>
<img src="https://drive.google.com/uc?id=1iLBCmZoWAxkVoN95mK_Aka97wDQ3-UBn "alt="image" width="550" height="500">
<figcaption><font size="+1"><b>Data types in the Lending club data: significant of them are floats</b></font></figcaption>
</center>
</figure>

<table border="1" class="dataframe">
<caption><b>Description of the lending club data</b></caption>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>int_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>0.0</td>
      <td>2.260668e+06</td>
      <td>2.260668e+06</td>
      <td>2.260668e+06</td>
      <td>2.260668e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>1.504693e+04</td>
      <td>1.504166e+04</td>
      <td>1.502344e+04</td>
      <td>1.309283e+01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>9.190245e+03</td>
      <td>9.188413e+03</td>
      <td>9.192332e+03</td>
      <td>4.832138e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>5.000000e+02</td>
      <td>5.000000e+02</td>
      <td>0.000000e+00</td>
      <td>5.310000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>8.000000e+03</td>
      <td>8.000000e+03</td>
      <td>8.000000e+03</td>
      <td>9.490000e+00</td>
    </tr>
  </tbody>
</table>
</section>
</h2>

<h2>
<section>
<b>Exploratory Data Analysis</b>
<a name="ref_sec_3">
<p>
The density of this dataset provided me a unique opportunity to explore it in a variety of ways. By grouping the data along various features I was able to explore it in various different ways. I first tried to understand how the customers of lending are distributed among different "grades" by simply counting it:

</p>
</a>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1--S7kMvp6UL1sHYpRB8ARs5vZWoMEeSC "alt="image" width="750" height="500">
<figcaption><font size="+1"><b>How the customers are distributed among different grades?</b></font></figcaption>
</center>
<p>It appears that the target customers are in mainly "B" and "C" category and not in the "A"</p>
</figure>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1-0Bi9vbjTNSCabpT--xAQUNO4ysIBWmu"alt="image_2" width="550" height="500">
<figcaption><font size="+1"><b> The sankey chart shows the relationship between the grades, home ownership and loan amounts</b></font></figcaption>
</center>
</figure>
<p>The thickness of the links between the customer grades and the home ownership status represents the count of the loans. It is obvious from here that the preferred customers are those which are currently under a mortgage plan followed by rent and own.</p>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1-0qHmhLmY0QziZ7QT-5EtFLGNuhU_tYx"alt="image_3" width="750" height="500">
<figcaption><font size="+1"><b>More concretely a $49.2\%$, $39.6\%$ and $11.2\%$  of the total customers are with mortgages, rent and own respectively. </b></font></figcaption>
</center>
</figure>

<figure>
<p><b>How is the fico score distributed?</b></p>
<center>
<img src="https://drive.google.com/uc?id=1-256bNKeI85qA573KttC7eQlxQ9rW8Q_"alt="image_4" width="550" height="500">
<figcaption><font size="+1"><b> Most of the lenders are in the range of 670~800 range. </b></font></figcaption>
</center>
</figure>

<figure>
<p><b>What is the relation between fico range, grades and loan?</b></p>
<center>
<img src="https://drive.google.com/uc?id=1-FTS03M8L1kc5mJOitDRRPlE_caSMcJx"alt="image_4" width="550">
<figcaption><font size="+1"><b> The sankey chart shows the relationship between the grades, fico range and loan amount counts-- as again 660-670 fico range is the preferred. </b></font></figcaption>
</center>
</figure>

<figure>
<p><b>Is the loan coming back to lending club?</b></p>
<center>
<img src="https://drive.google.com/uc?id=1-6ZSXhlU57tgjPLxmmmeHH00QFhnevgi"alt="image_6" width="550" height="500">
<figcaption><font size="+1"><b> Lots of fully paid loans!  </b></font></figcaption>
</center>
</figure>
<p>Lending club has a high number of customers either in the Fully paid and or Current range. This means that the company's model of lending the money is working well for them--the money which is going out is coming back too with some interest paid</p>
</section>
<p> <b>What is the employment history of the lenders?</b></p>
<figure>
<center>
<img src="https://drive.google.com/uc?id=1-Bt5mJh6Kh2aSLhnEOZxza_SPb084TCj"alt="image_7" width="550" height="500">
<figcaption><font size="+1"><b> Most of the lenders have the employment history of more than 10 years.  </b></font></figcaption>
</center>
</figure>
<p>By grouping the data with employment length shows that the lending club customers have signficant number of people who have a good employment history i.e. more than 10+ years.</p>

<p><b>What is the relationship between "mean" loan amount, installment and interest rate within each category?</b></p>
<figure>
<center>
<img src="https://drive.google.com/uc?id=1f5yXzKDQeWhMmstRn450Fm6Ga79QKOzn"alt="image_7" width="550" height="400">
<figcaption><font size="+1"><b> Bubble plot of "mean" loan amount vs mean "interest rate" features. Here the size of the bubble is related to the "mean" interest rate while the color of the bubbles represents the categories (A,B,C,D,E,F and G).  </b></font></figcaption>
</center>
</figure>
<p>Here I have grouped the data by grades and plotted them as bubble chart. Mean loan amount and mean installments show a linear behavior while the meant interest rate tend to grow sharply as we move from category B to G with an exception of category A which falls in between the categroy B and C.</p>

<p><b> At a more granular level how the interest rate is distributed within each category?</b></p>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1--vqbsXMJUCyCiJ2i13Kfe-vg8XHS6xV"alt="image_7" width="550" height="400">
<figcaption><font size="+1"><b> Interest rate distribution within each category of lenders.  </b></font></figcaption>
</center>
</figure>
<p> There is a very clear trend here among the interest within the categories $A < B < C < D$.</p>

<p><b>For which purpose the loan was taken?</b></p>
<figure>
<center>
<img src="https://drive.google.com/uc?id=1McN8Z7rtzWosBVpc4Rgoy3JU73ul6Wwz"alt="image_7" width="550" height="400">
<figcaption><font size="+1"><b> The "mean" loan is the highest for the small business followed by debt consolidation and so on.  </b></font></figcaption>
</center>
</figure>
<p> The "mean value" of the loan is highest for the small business purpose.</p>


<p><b>How the median of the loan amount, interest rate and annual income is distributed accross the states?</b></p>
<figure>
<center>
<img src="https://drive.google.com/uc?id=1-7hsSNysKZfrMjID16XQNhO8gW5HTOZN"alt="image_7" width="550" height="400">
<figcaption><font size="+1"><b> How the median loan is distributed amont different states of the US?  </b></font></figcaption>
</center>
</figure>


<figure>
<center>
<img src="https://drive.google.com/uc?id=1-Ea3XxvUr3QeV8eEYbH72M91jy8Ll83t"alt="image_7" width="550" height="400">
<figcaption><font size="+1"><b> How the median annual income is distributed amont different states of the US?  </b></font></figcaption>
</center>
</figure>

<p><b>How the charged off and the people who returned the loan are distributed accross the states?</b></p>
<figure>
<center>
<img src="https://drive.google.com/uc?id=1-HBiwX4AfdyTG83ej6VzVa42bhgS70wa"alt="image_7" width="550" height="400">
<figcaption><font size="+1"><b> How the charged off is distributed among different states--California leads the pack followed by New York.  </b></font></figcaption>
</center>
</figure>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1-L-9RJ0T2u4-rDnGVFoiufWEkhqpdp3b"alt="image_7" width="550" height="400">
<figcaption><font size="+1"><b> How the current loan status is distributed among different states--California leads the pack followed by Texas and New York.  </b></font></figcaption>
</center>
</figure>

<p><b>How the interest rate is distributed between different categories of loan status?</b></p>
</h2>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1-OF_F4CtH_h06fP4t2mZxkyaRZB0EPwD"alt="image_7" width="550" height="400">
<figcaption><font size="+1"><b> How the interest rate is distributed among different categories of loan status.  </b></font></figcaption>
</center>
</figure>

<p>The interest rate is lower for the payers category and higher for the charged off category on an average. </p>
<p><b>Uptill now I have plotted either the average values of the various features of the data in a variety of ways. Further to get a better estimate of the statistical features of the data I will make use of the violin plot for the interest rate on grade and states etc.</b></p>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1-RrlPtOkH7iTCMnEorSPMZykKmXFuSEE"alt="image_7" width="550" height="400">
<figcaption><font size="+1"><b> Violin plot of the interest among different grades of the customers.</b></font></figcaption>
</center>
</figure>

<p>Not only the grade 'A' has the minimal value of the median. The dispersion of interest is the least in category A. Clearly from the sheer interest rate point of view it is good to be in this categroy. The above plot further differentiate the data by their term period.</p>

<p><b>To see a rough value of the correlation between variables I will do a scatter matrix plot</b></p>
<figure>
<center>
<img src="https://drive.google.com/uc?id=1-S8gZOuPqAKDQJXEWKAQB45uKk6mQVyO"alt="image_7" width="750" height="600">
<figcaption><font size="+1"><b> Scatter plot of the various features of the data. </b></font></figcaption>
</center>
</figure>
<p>Is there are a strong monotonic relationship between interest rate and  loan amount? We can explore it using Spearmans Rank correlation coefficient.
More can be learned about it from this great resource <a href="https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide.php"> on Spearman's Rank-Order Correlation</a> Its definition is$\frac{1-6\sum d_{i}^2}{n(n^2-1)}$ here 'd' is the distance between the ranks of the data point.</p>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1-UIByWvKiBOqF4rhEsR6IsQ5vOJXwzb2"alt="image_7" width="550" height="400">
<figcaption><font size="+1"><b> Computed Spearman's correlation between loan amount, installment and interest rate.</b></font></figcaption>
</center>
</figure>
<p>Very high spearman rank correlation means as we increase the rank of one the other will increase montonically. This is true for loan and installment. Means these two variables are highly correlated and increase monotonically. While the other parameters doesn't increase monotonically.</p>
<h2>
<section>
<b>Training data development</b>
<a name="ref_sec_4">
<p>
<b>Removing the high correlated features</b>
<p>In this part I first removed the highly correlated variables. This is easier to do for the numerical data but not very straightforward for the categorical data. For the categorical data I used the cramers-V correlation and Thiel Uncertainty coefficient.</p>  

</a>
<figure>
<center>
<img src="https://drive.google.com/uc?id=1-_VynMd3STgWAm0BUxdF0M7I4-b-334w"alt="image_7" width="850" height="750">
<figcaption><font size="+1"><b> Heatmap of the correlation matrix. </b></font></figcaption>
</center>
</figure>

<p>How much correlated categorical features are? To compute the correlation between categorical value I took help of this article in which the author provides a link to his kaggle kernel <a href="https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9"></a>. From the basic equations of the pearson's correlation coefficient it can be gleaned that it is not designed to handle the categorical features. Some other method need to be defined. One such approach is the Cramer's V correlation which is a symmetrical measure and varies from 0-1.</p> 

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable_1</th>
      <th>variable_2</th>
      <th>Thiel U</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_status</td>
      <td>next_pymnt_d</td>
      <td>0.644694</td>
    </tr>
    <tr>
      <th>1</th>
      <td>loan_status</td>
      <td>last_pymnt_d</td>
      <td>0.602468</td>
    </tr>
    <tr>
      <th>2</th>
      <td>loan_status</td>
      <td>last_credit_pull_d</td>
      <td>0.278253</td>
    </tr>
    <tr>
      <th>3</th>
      <td>loan_status</td>
      <td>issue_d</td>
      <td>0.267395</td>
    </tr>
    <tr>
      <th>4</th>
      <td>loan_status</td>
      <td>title</td>
      <td>0.065845</td>
    </tr>
    <tr>
      <th>5</th>
      <td>loan_status</td>
      <td>initial_list_status</td>
      <td>0.033257</td>
    </tr>
    <tr>
      <th>6</th>
      <td>loan_status</td>
      <td>settlement_date</td>
      <td>0.031000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>loan_status</td>
      <td>debt_settlement_flag_date</td>
      <td>0.030703</td>
    </tr>
    <tr>
      <th>8</th>
      <td>loan_status</td>
      <td>settlement_status</td>
      <td>0.030590</td>
    </tr>
    <tr>
      <th>9</th>
      <td>loan_status</td>
      <td>debt_settlement_flag</td>
      <td>0.030425</td>
    </tr>
    <tr>
      <th>10</th>
      <td>loan_status</td>
      <td>sub_grade</td>
      <td>0.028570</td>
    </tr>
    <tr>
      <th>11</th>
      <td>loan_status</td>
      <td>grade</td>
      <td>0.026454</td>
    </tr>
    <tr>
      <th>12</th>
      <td>loan_status</td>
      <td>disbursement_method</td>
      <td>0.019841</td>
    </tr>
    <tr>
      <th>13</th>
      <td>loan_status</td>
      <td>sec_app_earliest_cr_line</td>
      <td>0.018732</td>
    </tr>
    <tr>
      <th>14</th>
      <td>loan_status</td>
      <td>earliest_cr_line</td>
      <td>0.018138</td>
    </tr>
    <tr>
      <th>15</th>
      <td>loan_status</td>
      <td>application_type</td>
      <td>0.016498</td>
    </tr>
    <tr>
      <th>16</th>
      <td>loan_status</td>
      <td>verification_status_joint</td>
      <td>0.015232</td>
    </tr>
    <tr>
      <th>17</th>
      <td>loan_status</td>
      <td>verification_status</td>
      <td>0.007668</td>
    </tr>
    <tr>
      <th>18</th>
      <td>loan_status</td>
      <td>payment_plan_start_date</td>
      <td>0.004565</td>
    </tr>
    <tr>
      <th>19</th>
      <td>loan_status</td>
      <td>hardship_end_date</td>
      <td>0.004536</td>
    </tr>
    <tr>
      <th>20</th>
      <td>loan_status</td>
      <td>hardship_start_date</td>
      <td>0.004467</td>
    </tr>
    <tr>
      <th>21</th>
      <td>loan_status</td>
      <td>hardship_status</td>
      <td>0.004404</td>
    </tr>
    <tr>
      <th>22</th>
      <td>loan_status</td>
      <td>hardship_reason</td>
      <td>0.003484</td>
    </tr>
    <tr>
      <th>23</th>
      <td>loan_status</td>
      <td>hardship_loan_status</td>
      <td>0.003372</td>
    </tr>
    <tr>
      <th>24</th>
      <td>loan_status</td>
      <td>hardship_type</td>
      <td>0.003192</td>
    </tr>
    <tr>
      <th>25</th>
      <td>loan_status</td>
      <td>purpose</td>
      <td>0.003024</td>
    </tr>
    <tr>
      <th>26</th>
      <td>loan_status</td>
      <td>emp_length</td>
      <td>0.002125</td>
    </tr>
    <tr>
      <th>27</th>
      <td>loan_status</td>
      <td>home_ownership</td>
      <td>0.001983</td>
    </tr>
    <tr>
      <th>28</th>
      <td>loan_status</td>
      <td>hardship_flag</td>
      <td>0.001689</td>
    </tr>
    <tr>
      <th>29</th>
      <td>loan_status</td>
      <td>pymnt_plan</td>
      <td>0.001344</td>
    </tr>
  </tbody>
  <caption><b>Thiel U coefficient table</b></caption>
</table>

<p>From Cramers V correlation it can be said that some of the features are highly correlated with each other (grade and sub_grade) while the Theil U coefficient inform about the importance of a particular feature which indicates directly towards the target variable. So first I will keep those object type variables which have good enough information about the target variables and then remove the object type variables which are highly correlated among themselves.</p>
<p><b>Encoding the categorical features</b></p>
<p>Some of the object features are of very high cardinality. one hot encoding will make a feature explosion. The following link and library mentions about some ways to deal with features with very high cardinality. <a href="http://contrib.scikit-learn.org/category_encoders/index.html"></a>. I can use hashing but the hashing algorithm may put different categories in the same group--might affect the target negatively. Other way around is the target encoding which is explained quite well in the following link--<a href="https://maxhalford.github.io/blog/target-encoding/"></a>

I could have used the mean encoding. However, I decide to follow the Laplace smoothing of the local mean. It is because if there is a very low occurence of a certain instance of a feature then its local mean cannot be trusted. Then a more heavy emphasis should be given on the global mean.</p>

<table border="1" class="dataframe">
<caption></caption>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>less_than_200_counts_categories</th>
      <th>greater_than_200_counts_categories</th>
      <th>unique_category</th>
      <th>ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>grade</th>
      <td>1.0</td>
      <td>7.0</td>
      <td>8</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>emp_length</th>
      <td>0.0</td>
      <td>12.0</td>
      <td>12</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>verification_status</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>4</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>issue_d</th>
      <td>14.0</td>
      <td>126.0</td>
      <td>140</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>purpose</th>
      <td>1.0</td>
      <td>14.0</td>
      <td>15</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>earliest_cr_line</th>
      <td>235.0</td>
      <td>520.0</td>
      <td>755</td>
      <td>2.212766</td>
    </tr>
    <tr>
      <th>initial_list_status</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>last_pymnt_d</th>
      <td>32.0</td>
      <td>105.0</td>
      <td>137</td>
      <td>3.281250</td>
    </tr>
    <tr>
      <th>next_pymnt_d</th>
      <td>104.0</td>
      <td>3.0</td>
      <td>107</td>
      <td>0.028846</td>
    </tr>
    <tr>
      <th>last_credit_pull_d</th>
      <td>56.0</td>
      <td>86.0</td>
      <td>142</td>
      <td>1.535714</td>
    </tr>
    <tr>
      <th>application_type</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>verification_status_joint</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sec_app_earliest_cr_line</th>
      <td>462.0</td>
      <td>202.0</td>
      <td>664</td>
      <td>0.437229</td>
    </tr>
    <tr>
      <th>hardship_status</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>disbursement_method</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>debt_settlement_flag</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>debt_settlement_flag_date</th>
      <td>56.0</td>
      <td>28.0</td>
      <td>84</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>settlement_status</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>settlement_date</th>
      <td>61.0</td>
      <td>30.0</td>
      <td>91</td>
      <td>0.491803</td>
    </tr>
  </tbody>
</table>
</p>

<p>It means there are 14 columns with less than 200 counts and all the other columns have higher count. I decided to work with the local mean where the count is higher than 200 else I decided to divert the mean towards the global mean.</p>
<p><b>Feature leakage</b>:-- leakage is a situation in which a model is built using data which is not available at the time the model will be used to make a prediction. Considering this and after checking the description of the data in the lending club data dictionary--I will drop these features <i><font color="blue">('issue_d_encoded','total_rec_late_fee','debt_settlement_flag_encoded','settlement_status_encoded','settlement_amount','settlement_date_encoded','debt_settlement_flag_date_encoded','settlement_amount','last_pymnt_d_encoded','settlement_date_encoded','next_pymnt_d_encoded','recoveries','out_prncp','last_pymnt_amnt','last_credit_pull_d_encoded','last_fico_range_high','total_pymnt')</font></i> as these features may not be available at the time of classification and or may be too indicative of the target value. For example the recoveries feature is the "post charge off recovery"---this is only available after a charge off has occured and shouldn't be included in the model development.</p>
</section>
</h2>

<h2>
<section>
<b>Modeling</b>
<a name="ref_sec_5">
<p>
<b>Outlier removal</b> </p>
<p>
Outliers can create some problems for the machine learning algorithms further down the line and lead to incorrect conclusions. If there is any specially in the float data then I will remove it using the quantile method.
</p>
</a>
<figure>
<center>
<img src="https://drive.google.com/uc?id=1KJ2tcxH-R76iTeCtBJyleChRDeeWVqZ0"alt="image_7" width="950" height="550">
<figcaption><font size="+1"><b> Raw loan vs the log transformed data. </b></font></figcaption>
</center>
</figure>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>annual_inc</th>
      <th>loan_amnt</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24659</td>
      <td>8700000.0</td>
      <td>14000.0</td>
      <td>debt_consolidation</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29400</td>
      <td>6000000.0</td>
      <td>18500.0</td>
      <td>debt_consolidation</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38473</td>
      <td>7000000.0</td>
      <td>25000.0</td>
      <td>debt_consolidation</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40533</td>
      <td>9000000.0</td>
      <td>11000.0</td>
      <td>debt_consolidation</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48075</td>
      <td>8500021.0</td>
      <td>12000.0</td>
      <td>credit_card</td>
    </tr>
    <tr>
      <th>5</th>
      <td>75685</td>
      <td>5000010.0</td>
      <td>20000.0</td>
      <td>credit_card</td>
    </tr>
    <tr>
      <th>6</th>
      <td>85757</td>
      <td>8253000.0</td>
      <td>30000.0</td>
      <td>debt_consolidation</td>
    </tr>
    <tr>
      <th>7</th>
      <td>128385</td>
      <td>8121180.0</td>
      <td>5000.0</td>
      <td>debt_consolidation</td>
    </tr>
    <tr>
      <th>8</th>
      <td>132773</td>
      <td>7600000.0</td>
      <td>10000.0</td>
      <td>home_improvement</td>
    </tr>
    <tr>
      <th>9</th>
      <td>205670</td>
      <td>6000000.0</td>
      <td>35000.0</td>
      <td>debt_consolidation</td>
    </tr>
    <tr>
      <th>10</th>
      <td>209413</td>
      <td>6000000.0</td>
      <td>4475.0</td>
      <td>home_improvement</td>
    </tr>
    <tr>
      <th>11</th>
      <td>217721</td>
      <td>8900060.0</td>
      <td>10550.0</td>
      <td>debt_consolidation</td>
    </tr>
    <tr>
      <th>12</th>
      <td>230972</td>
      <td>9500000.0</td>
      <td>24000.0</td>
      <td>credit_card</td>
    </tr>
    <tr>
      <th>13</th>
      <td>315219</td>
      <td>7000000.0</td>
      <td>7500.0</td>
      <td>car</td>
    </tr>
    <tr>
      <th>14</th>
      <td>400638</td>
      <td>8706582.0</td>
      <td>8000.0</td>
      <td>credit_card</td>
    </tr>
    <tr>
      <th>15</th>
      <td>444152</td>
      <td>8020871.0</td>
      <td>35000.0</td>
      <td>debt_consolidation</td>
    </tr>
    <tr>
      <th>16</th>
      <td>461167</td>
      <td>8365188.0</td>
      <td>10000.0</td>
      <td>debt_consolidation</td>
    </tr>
    <tr>
      <th>17</th>
      <td>473702</td>
      <td>6032121.0</td>
      <td>25000.0</td>
      <td>debt_consolidation</td>
    </tr>
    <tr>
      <th>18</th>
      <td>474214</td>
      <td>7500000.0</td>
      <td>35000.0</td>
      <td>credit_card</td>
    </tr>
    <tr>
      <th>19</th>
      <td>539807</td>
      <td>10999200.0</td>
      <td>5000.0</td>
      <td>major_purchase</td>
    </tr>
  </tbody>
  <caption><b>Dataframe where the income and purpose of the loan are put togethere side by side. Only those lenders are selected who earn in $7$ figures. Quite a few people are asking for "debt consolidation".</b></caption>
</table>
<p>Log transforming present a clear case that there are too many entries above and below the mean--specially there are quite a few values close to million. This can't be a human error. Further grouping the data by loan and purpose I can see that a person earning a $9$ figure salary is taking loan for debt consolidation(?). This looks a little off but it cannot stop this person for taking loan as anyone can take loan for any purpose. And there are quite a few entries in the dataset where the lender earns in millions and is asking for a loan for debt consolidation.</p>
<p><b>Model training</b>
Since, this is a classification problem and the way to categorize the correctness of the trained model is rather subjective and is based on the idea of confusion matrix.
However, the following measures are readily available for the classification problem:</p>
<figure>
<center>
<img src="https://drive.google.com/uc?id=1n97eKj1HvOdE-cg2OTc1YlvT0X8VmjcV"alt="image_7" width="550" height="400">
<figcaption><font size="+1"><b> Confusion matrix.  </b></font></figcaption>
</center>
</figure>

<ul>
<li>For example Accuracy = $\frac{\text{Total number of correct predictions}}{\text{Total predictions made}}$</li>
<li> Precision = $\frac{\text{True positive}}{\text{True positive + False positive}}$</li>
<li>Recall = $\frac{\text{True positive}}{\text{True positive + False negative}}$</li>
<li>F1 score= $\frac{\text{2* Precision*Recall}}{\text{Precision+Recall}}$</li>
<li> MCC = $\frac{TP*TN-FP*FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN}}$. </li>
</ul>
<p>Considering this I will optimize for Recall as I don't want someone to be categorized as non defaulter even though he/she is going to do a default.</p>
<ul>
<li> <b>Logistic regression</b> </li> For solving the classification problem I will start with the simplest classifier aka the logistic regression. 
</ul>
<figure>
<center>
<img src="https://drive.google.com/uc?id=1-AMMp37Oe98DkkX7mEdilCjsJAG0H6Z_"alt="image_7" width="450"/> <img src="https://drive.google.com/uc?id=1AkCK42jQfrliaknd-rArzERiErbJXOR0"alt="image_7" width="450"/>
<figcaption><font size="+1"><b> Receiver operating characteristic (ROC) and precision recall curve for a tuned (hyperparameter tunning) logistic regression model.  </b></font></figcaption>
</center>
</figure>

<center>
<figure>
<img src="https://drive.google.com/uc?id=1-Oh3BRzDM4i169-FdI6UNqEoG3zM-EVk"alt="image_7" width="550" > 
<figcaption><font size="+1"><b> Confusion matrix for the tuned logistic regression model.  </b></font></figcaption>
</figure>
</center>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1-1RcIMmTwBcXgdIThu9civShGlHS3QDK"alt="image_7" width="550" > 
<figcaption><font size="+1"><b> Which feature got the highest significance in the tuned logistic regression model </b></font></figcaption>
</center>
</figure>
<p>Logistic regression puts the highest importance on the annual income followed by number of derogatory public records--all of which makes complete sense. However, I am not satisfied by its higher recall value considering this I will try some ensemble learning techniques.</p>
<ul>
<li> <b>Light gradient boosting</b> </li>
</ul>
<figure>

<center>
<img src="https://drive.google.com/uc?id=1-EOkVRfaH_kyBZHFACutCGfDHosE6jei"alt="image_7" width="450"/> <img src="https://drive.google.com/uc?id=1Ij9u6yYoBu5F5gy8q_m_n16Fwh6Da5b8"alt="image_7" width="450"/>
<figcaption><font size="+1"><b> Receiver operating characteristic (ROC) and precision recall curve for a tuned (hyperparameter tunning) light gradient boosting model.  </b></font></figcaption>
</center>
</figure>

<center>
<figure>
<img src="https://drive.google.com/uc?id=1-SElO4DWxPAtFS2chO7sunNkS83aIFVx"alt="image_7" width="550" > 
<figcaption><font size="+1"><b> Confusion matrix for the tuned light gradient boosting model.  </b></font></figcaption>
</figure>
</center>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1-MyaYOb__L28JdADNGgxRP_zXKJ-d8Fn"alt="image_7" width="550" > 
<figcaption><font size="+1"><b> Which feature got the highest significance in the tuned light gradient boosting model. </b></font></figcaption>
</center>
</figure>

<p>Surprisingly this model performs poorly than the logistic regression. It has a higher false positive rate while the true positive rate is also low. Meaning this must be having a lower value of recall.</p>


<ul>
<li> <b>Xtreme gradient boosting</b> </li>
</ul>


<figure>
<center>
<img src="https://drive.google.com/uc?id=1-UAoBAWSX2PZH7V293nkDobtLlin0kDx"alt="image_7" width="450"/> <img src="https://drive.google.com/uc?id=1--QYRt3mG_UiQG3kkQVQZBO-JyLJZVnD"alt="image_7" width="450"/>
<figcaption><font size="+1"><b> Receiver operating characteristic (ROC) and precision recall curve for a tuned (hyperparameter tunning) XGBOOST model.  </b></font></figcaption>
</center>
</figure>


<figure>
<center>
<img src="https://drive.google.com/uc?id=1-WK2h_qw0lqNgOtAeohoY-lUw-GXju72"alt="image_7" width="550"> 
<figcaption><font size="+1"><b> Confusion matrix for XGBOOST model.  </b></font></figcaption>
</center>
</figure>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1dG02hpM7jfBzwevodA2gbGo351oeRcFK"alt="image_7" width="550"> 
<figcaption><font size="+1"><b> Which feature got the highest importance in XGBOOST.  </b></font></figcaption>
</center>

</figure>
<p>XGBOOST has a very low value of false negative and a higher true positive rate. This means its recall should be better.</p>


</figure>

<ul>
<li> <b>Catboost model</b> </li>
</ul>
<p>At the end I tried categorical boosting without the target encoding performed on the dataset. This is because the catboost model is specifically designed to work with categorical data.</p>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1Q2uFgZv003bawjnUgPeP6OGLGEHncZZv"alt="image_7" width="450"/> <img src="https://drive.google.com/uc?id=1-j8v2h5WkSpJDVdL4mk2BAtdBcsZo69z"alt="image_7" width="450"/>
<figcaption><font size="+1"><b> Receiver operating characteristic (ROC) and precision recall curve for a tuned (hyperparameter tunning) Catboost model.  </b></font></figcaption>
</center>
</figure>
<p>The recall for tuned catboost model tend to be very low. In fact it performed poorer than the Xgboost model. This need some fine tuning and reconsideration of the feature cleanup.</p>

<ul><li><b>Explaining the affect of the features on the model outcome</b></li></ul>

<p>How much has each feature value contributed to the prediction compared to the average prediction?
The answer to this question lies in the Shap values--which essentially tells how much a change in feature will drive the output away from the average prediction. A typical example of the house price prediction is given here <a href="https://christophm.github.io/interpretable-ml-book/shap.html"></a> which explains the idea quite well. For linear model the driver which will move the output away from average will be the weight times the feature value but for non-linear model this is not straightforward to answer. The Shap values are model agnostic and exactly explains the same. SHAP values has their origin in the game theory. The “game” here is the prediction task for a single instance of the dataset. The “players” are the feature values of the instance that collaborate to play the game (predict a value)--i.e. the loan is going to get charged off or not. In short Shapley values correspond to the contribution of each feature towards pushing the prediction away from the expected value of the model outcome. The above analysis was for shap values for each feature of every observation, it is possible to get a global interpretation using Shapley values by looking at it in a combined form.
</p>

<figure>
<center>
<img src="https://drive.google.com/uc?id=1PTy9D4FsxbQWEBqpDB0DB9xCXWVGq2vt"alt="image_7" width="450">
<figcaption><font size="+1"><b> SHAP values for the XGBOOST model. </b></font></figcaption>
</center>
</figure>
<p>Target encoded grade has the highest effect on the model output. This makes sense as I saw in the Sankey chart that there is a good correlation between the grades, house ownership and the loans. This means that the lending club tend to trust people which falls in the better category--which intuitively makes sense.</p>
</section>
</h2>
<h2>
<section>

<b>Conclusion</b>
<a name="ref_sec_6">
<p>
A significant number of features in the current dataset are not indicative of the defaulters. A good feature engineering can lead to a simpler model--for example logistic regression. By performing the target encoding I was able to achieve decent result both with logistic regression and Xgboost models. Catboost is supposed to perform better however its abysmal recall value is not acceptable. In future I will try to improve the performance of catboost model by a more careful consideration. In the end the model results were explained using a metric derived from game theory i.e. the shap value.  
</p>
</a>
</section>
</h2>

