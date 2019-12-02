Here we'll use mutual information to compute correlations between different variables measured by a survey. The focus of the survey is on disability in a humanitarian context. 
In this case, the dataset is a survey on Syrian refugees living in camps in Jordan and Lebanon. You can read the report (on the Jordan subset of the data) [here](https://reliefweb.int/report/jordan/removing-barriers-path-towards-inclusive-access-disability-assessment-among-syrian)

```python
from mutual_information import *
%load_ext autoreload
%autoreload 2
```

# let's load some data

For now, we are going to analyze the correlations between the variables (questions) measured in the survey only for adults.

```python
data = pd.read_csv('./Data/data_adult.csv')
```

# what does the data look like? 

Lets look at the first few rows in the data.


```python
data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#loc_country</th>
      <th>#loc_camp</th>
      <th>#access+shelter_Does your family have a shelter or house with electricity, easy to move around, in a location where you feel safe? Answer below questions by Yes or NoIf no, what is your concern?</th>
      <th>#access+shelter_Yes I have shelter/house but no stable electricity</th>
      <th>#access+shelter_Yes I have shelter/house but not easy to move around for one or some of family members</th>
      <th>#access+shelter_yes I have shelter /house but  fear of attack or harassment around my place</th>
      <th>#access+shelter_Yes I have shelter/house but fear of fear to getting harm or injuries around my place</th>
      <th>#barriers+shelter_If no, what is your concern?</th>
      <th>#access+medical_If there was a medical need, were you or any of your family members able to access medical services at hospitals/clinics in the last six months?  If no, why?</th>
      <th>#access+medical_Did you access to medical services when needed?</th>
      <th>#barriers+medical_I do not know where the service or support is available, or, who can help?</th>
      <th>#barriers+medical_I got some information but could not read or understand?</th>
      <th>#barriers+medical_I do not have documents to access services (Specify types of documents: eg UNHCR card, visa, ID)?</th>
      <th>#barriers+medical_Safety fears for movement outside home (attack, harassment,..)?</th>
      <th>#barriers+medical_Safety fears for movement outside home (harm, injuries  arrested..)?</th>
      <th>#barriers+medical_Services are not available?</th>
      <th>#barriers+medical_Services are too expensive ?</th>
      <th>#barriers+medical_Services are far away and transportation is not available?</th>
      <th>#barriers+medical_Services are far away and transportation is too expensive?</th>
      <th>#barriers+medical_Services are far away and transportation is not accessible?</th>
      <th>#barriers+medical_Services are delivered in places that are not accessible?</th>
      <th>#barriers+medical_Services are delivered in places that are not gender sensitive (not comfortable for women or men)?</th>
      <th>#barriers+medical_Staff are not supportive and/or do not know how to communicate with me/my family?</th>
      <th>#barriers+medical_Services are not meeting my/my family’s specific needs?</th>
      <th>#barriers+medical_Others (specify)?</th>
      <th>#barriers+medical_Others (specify)</th>
      <th>#barriers+medical_Among those mentioned as reasons, what will be the most important issue which, if solved, will help you access services? Select one please</th>
      <th>#access+water_Does your family have enough safe  water from reliable sources for drinking, cooking, cleaning and personal hygiene?</th>
      <th>#barriers+water_I do not know where the service or support is available, or, who can help?6</th>
      <th>#barriers+water_I got some information but could not read or understand?7</th>
      <th>#barriers+water_I do not have documents to access services (Specify types of documents: (eg UNHCR card, visa, ID)?</th>
      <th>#barriers+water_Safety fears for movement outside home (attack, harassment, arrested)?</th>
      <th>#barriers+water_Safety fears for movement outside home (harm, injuries)?</th>
      <th>#barriers+water_Services are not available?8</th>
      <th>#barriers+water_Services are too expensive ?9</th>
      <th>#barriers+water_Services are far away and transportation is not available?10</th>
      <th>#barriers+water_Services are far away and transportation is too expensive?11</th>
      <th>#barriers+water_Services are far away and transportation is not accessible?12</th>
      <th>#barriers+water_Services are delivered in places that are not accessible?13</th>
      <th>#barriers+water_Services are delivered in places that are not gender sensitive (not comfortable for women or men)?14</th>
      <th>#barriers+water_Staff are not supportive and/or do not know how to communicate with me/my family?15</th>
      <th>#barriers+water_Services are not meeting my/my family’s specific needs?16</th>
      <th>#barriers+water_Others (specify)?17</th>
      <th>#barriers+water_Others (specify)18</th>
      <th>#barriers+water_Among those mentioned as reasons, what will be the most important issue which, if solved, will help you access services? Select one please20</th>
      <th>#barriers+latrine_Is it clean?</th>
      <th>#access+latrine_Is it accessible for one or some of family members?</th>
      <th>#barriers+latrine_Is it available all the time to use freely?</th>
      <th>#barriers+latrine_Fear of attack, harassment?</th>
      <th>#barriers+latrine_fear of  harm or injuries?</th>
      <th>...</th>
      <th>#barriers+cash_I got some information but could not read or understand?38</th>
      <th>#barriers+cash_I do not have documents to access services (Specify types of documents: (eg UNHCR card, visa, ID)?39</th>
      <th>#barriers+cash_Safety fears for movement outside home (attack, harassment, arrested)?40</th>
      <th>#barriers+cash_Safety fears for movement outside home (harm, injuries)?41</th>
      <th>#barriers+cash_Services are not available?42</th>
      <th>#barriers+cash_Services are too expensive ?43</th>
      <th>#barriers+cash_Services are far away and transportation is not available?44</th>
      <th>#barriers+cash_Services are far away and transportation is too expensive?45</th>
      <th>#barriers+cash_Services are far away and transportation is not accessible?46</th>
      <th>#barriers+cash_Services are delivered in places that are not accessible?47</th>
      <th>#barriers+cash_Services are delivered in places that are not gender sensitive (not comfortable for women or men)?48</th>
      <th>#barriers+cash_Staff are not supportive and/or do not know how to communicate with me/my family?49</th>
      <th>#barriers+cash_Services are not meeting my/my family’s specific needs?50</th>
      <th>#barriers+cash_Other reasons (specify)?</th>
      <th>#barriers+cash_Aid was cut off from the family</th>
      <th>#barriers+cash_They registered and waited for a response</th>
      <th>#access+cash+income_What is your household total cash income in the past month?</th>
      <th>#access+cash+income_What is your household total cash income in the past month?53</th>
      <th>#access+cash+income_What is your total amount of debt accumulated since your arrival to (${country}) up to now</th>
      <th>#access+cash+income_What is your total amount of debt accumulated since your arrival to (${country}) up to now54</th>
      <th>#personal_Age</th>
      <th>#personal_gender</th>
      <th>#personal_employed</th>
      <th>#wgq+ss_Do you have difficulty seeing, even when wearing glasses?</th>
      <th>#wgq+ss_Do  you  have  difficulty  hearing,  even  when  using a  hearing aid?</th>
      <th>#wgq+es_How often do you use your hearing aid(s)?</th>
      <th>#wgq+es_Do you use any equipment or receive help for getting around?</th>
      <th>#wgq+ss_Do you have difficulty walking or climbing steps?</th>
      <th>#wgq+es_Do you use Cane or walking stick?</th>
      <th>#wgq+es_Do you use Walker or Zimmer frame?</th>
      <th>#wgq+es_Do you use Crutches?</th>
      <th>#wgq+es_Do you use Wheelchair?</th>
      <th>#wgq+es_Do you use Artificial limb (leg/foot)?</th>
      <th>#wgq+es_Do you use Someone’s assistance?</th>
      <th>#wgq+es_Do you use Other equipment?(please specify)</th>
      <th>#wgq+es_Please specify the other equipment</th>
      <th>#wgq+ss_Using your usual language, do you have difficulty communicating, for example understanding or being understood?</th>
      <th>#wgq+es_Do you use sign language?</th>
      <th>#wgq+ss_Do you have difficulty remembering or concentrating?</th>
      <th>#wgq+ss_Do you have difficulty with self care, such as washing all over or dressing?</th>
      <th>#wgq+en_Do you have difficulty raising a 2 litre jug of water or soda from waist to eye level?</th>
      <th>#wgq+en_Do you have difficulty using your hands and fingers, such as picking up small objects, for example, a button or pencil, or opening or closing containers or bottles?</th>
      <th>#wgq+en_How often do you feel worried, nervous or anxious?</th>
      <th>#wgq+en_Thinking about the last time you felt  worried, nervous or anxious, how would you  describe the level of these feelings?</th>
      <th>#wgq+en_How often do you feel depressed?</th>
      <th>#wgq+en_Thinking about the last time you felt depressed, how depressed did you feel?</th>
      <th>#wgq+es_In the past 3 months, how often did you feel very tired or exhausted?</th>
      <th>#wgq+es_Thinking about the last time you felt very tired or exhausted, how long did it last?</th>
      <th>#wgq+es_Thinking about the last time you felt this way, how would you describe the level of tiredness?</th>
      <th>#disability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 120 columns</p>
</div>



You may notice that the column names start with a hashtag e.g. "#access+shelter_Yes I have shelter/house but no stable electricity". This hash tag will help us categorize the variables into useful categories (e.g. #access+shelter tags are questions about access to shelter) and help visualize them in a network.

The data contains missing values (NaN), and the answers to the questions have been recoded to numerical values

# Correlations : let's compute Mutual Information! 

To analyze correlations between variables, we'll use a very useful measure of statistical dependence between variables called [Mutual Information](https://en.wikipedia.org/wiki/Mutual_information#cite_note-magerman-19)

This measure has its origins in the work of [Claude Shannon](https://en.wikipedia.org/wiki/Claude_Shannon) and is has deep relationships with both basic science (statistical physics) and data science and machine learning.

We're going to use a normalized version of Mutual Information here (called [Information Quality Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0169743916304907?via%3Dihub)). This means the mutual information values will lie between 0.0 and 1.0.


```python
# OPTIONS AVAILABLE : 
# MI_type specifies the exact mutual information measure to use
# na_remove : removes all rows from the data where there are missing values in the pair of variables being compared
# na_thresh : how many non-missing value rows in the pair of variables are needed ?
# MI_thresh : the mutual information needs to be at least this much to appear in the graph

mi_graph = MIGraphBuilder(MI_type = 'IQR', na_remove = True, na_thresh = 200, MI_thresh = 0.008)
```

# plot the network

We'll only look at sub-graph where we look at relationships between [Washington Group Questions](http://www.washingtongroup-disability.com/) and access (and barriers to access) to cash assistance. We'll also include variables about personal information (age, gender, employment etc.) and an indicator of disability that is derived from the Washington Group Questions (WGQ). The WGQs are widely used to assess the difficulties people have in performing everyday basic activities such as reading, hearing, walking, taking care of themselves, remembering things etc.  


```python
# lets first compute the Mutual Information
E, _ = mi_graph.compute_graph(data, return_all=True)
```

     ..... MI graph computed in 13.03 secs ...... 



```python
# now lets sub-select a part of the graph
E = mi_graph.select_nodes(select_group = ['#wgq', '#barriers+food', '#access+food', '#disability', '#barriers+cash', '#access+cash', '#personal'], only_these=True)
```

How many nodes are there in this sub-graph ? 


```python
len(E.keys())
```




    1542



To plot this, we need to use the excellent python library [networkX](https://networkx.github.io/documentation/stable/index.html) 


```python
Gr, pos, weights = mi_graph.setup_nx_graph(E)
```

     ..... NX graph computed in 0.03 secs ...... 


now lets draw it !


```python
mi_graph.draw_graph(Gr, pos, weights, link_weight=2., fig_size = (30,20))
```

     ..... NX graph plotted in 2.10 secs ..... 



![png](output_17_1.png)


Here are the top 20 pairs of variables based on the mutual information between them. In the table you also find the p-value according to [G-test](https://en.wikipedia.org/wiki/G-test)


```python
df = mi_graph.save_top_mi_values(E, topK=20)
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node1</th>
      <th>node1name</th>
      <th>node2</th>
      <th>node2name</th>
      <th>mutual_info</th>
      <th>p-value</th>
      <th>sample_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>71</td>
      <td>#barriers+cash_I do not have documents to acce...</td>
      <td>0.570677</td>
      <td>0.0</td>
      <td>374</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56</td>
      <td>#barriers+food_Services are not available?25</td>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>0.561433</td>
      <td>0.0</td>
      <td>954</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>#barriers+food_I do not have documents to acce...</td>
      <td>71</td>
      <td>#barriers+cash_I do not have documents to acce...</td>
      <td>0.540571</td>
      <td>0.0</td>
      <td>374</td>
    </tr>
    <tr>
      <th>3</th>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>68</td>
      <td>#access+cash_Does your family receive any cash...</td>
      <td>0.531209</td>
      <td>0.0</td>
      <td>954</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>#barriers+food_I do not have documents to acce...</td>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>0.508244</td>
      <td>0.0</td>
      <td>954</td>
    </tr>
    <tr>
      <th>5</th>
      <td>56</td>
      <td>#barriers+food_Services are not available?25</td>
      <td>74</td>
      <td>#barriers+cash_Services are not available?42</td>
      <td>0.489562</td>
      <td>0.0</td>
      <td>374</td>
    </tr>
    <tr>
      <th>6</th>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>74</td>
      <td>#barriers+cash_Services are not available?42</td>
      <td>0.387520</td>
      <td>0.0</td>
      <td>374</td>
    </tr>
    <tr>
      <th>7</th>
      <td>56</td>
      <td>#barriers+food_Services are not available?25</td>
      <td>68</td>
      <td>#access+cash_Does your family receive any cash...</td>
      <td>0.328470</td>
      <td>0.0</td>
      <td>954</td>
    </tr>
    <tr>
      <th>8</th>
      <td>57</td>
      <td>#barriers+food_Services are too expensive ?26</td>
      <td>75</td>
      <td>#barriers+cash_Services are too expensive ?43</td>
      <td>0.319383</td>
      <td>0.0</td>
      <td>374</td>
    </tr>
    <tr>
      <th>9</th>
      <td>51</td>
      <td>#barriers+food_I do not know where the service...</td>
      <td>69</td>
      <td>#barriers+cash_I do not know where the service...</td>
      <td>0.307482</td>
      <td>0.0</td>
      <td>374</td>
    </tr>
    <tr>
      <th>10</th>
      <td>58</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>76</td>
      <td>#barriers+cash_Services are far away and trans...</td>
      <td>0.303753</td>
      <td>0.0</td>
      <td>374</td>
    </tr>
    <tr>
      <th>11</th>
      <td>112</td>
      <td>#wgq+en_How often do you feel worried, nervous...</td>
      <td>114</td>
      <td>#wgq+en_How often do you feel depressed?</td>
      <td>0.302617</td>
      <td>0.0</td>
      <td>3866</td>
    </tr>
    <tr>
      <th>12</th>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>75</td>
      <td>#barriers+cash_Services are too expensive ?43</td>
      <td>0.282430</td>
      <td>0.0</td>
      <td>374</td>
    </tr>
    <tr>
      <th>13</th>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>69</td>
      <td>#barriers+cash_I do not know where the service...</td>
      <td>0.275629</td>
      <td>0.0</td>
      <td>374</td>
    </tr>
    <tr>
      <th>14</th>
      <td>97</td>
      <td>#wgq+ss_Do you have difficulty walking or clim...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.273227</td>
      <td>0.0</td>
      <td>3866</td>
    </tr>
    <tr>
      <th>15</th>
      <td>59</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>77</td>
      <td>#barriers+cash_Services are far away and trans...</td>
      <td>0.271819</td>
      <td>0.0</td>
      <td>374</td>
    </tr>
    <tr>
      <th>16</th>
      <td>60</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>76</td>
      <td>#barriers+cash_Services are far away and trans...</td>
      <td>0.267379</td>
      <td>0.0</td>
      <td>374</td>
    </tr>
    <tr>
      <th>17</th>
      <td>93</td>
      <td>#wgq+ss_Do you have difficulty seeing, even wh...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.259761</td>
      <td>0.0</td>
      <td>562</td>
    </tr>
    <tr>
      <th>18</th>
      <td>51</td>
      <td>#barriers+food_I do not know where the service...</td>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>0.253518</td>
      <td>0.0</td>
      <td>954</td>
    </tr>
    <tr>
      <th>19</th>
      <td>65</td>
      <td>#barriers+food_Others (specify)?34</td>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>0.245082</td>
      <td>0.0</td>
      <td>879</td>
    </tr>
  </tbody>
</table>
</div>



How can we identify very "important" variables ? One way of looking at it is through the lens of Graph Theory. If a variable is highly correlated with several variables, which in turn are correlated with others, this variable has a higher value "centrality" in the graph. Let's compute "eigenvector" centrality : this is based on the graph adjacency matrix's principal eigenvector.


```python
C = mi_graph.compute_centrality(Gr, centrality_type='between')
```

the variable with the higest centrality is : 


```python
print(' .... Variable name : %s , centrality %.3f ....' %(data.columns[C[0][0]], C[0][1]))
```

     .... Variable name : #barriers+food_Among those mentioned as reasons, what will be the most important issue which, if solved, will help you access services? Select one please36 , centrality 0.089 ....


# What about correlations between variables based only on people with disability ? 


```python
# let's make a new graph!
data_disability = data.loc[data['#disability']==1, :]
```


```python
data_disability.shape
```




    (1140, 120)




```python
# instantiate the graph builder object
mi_graph_disability = MIGraphBuilder(MI_type = 'IQR', na_remove = True, na_thresh = 200, MI_thresh = 0.02)
# compute edges
E_disability, _ = mi_graph_disability.compute_graph(data_disability, return_all=True)
# sub-select edges
E_disability = mi_graph_disability.select_nodes(select_group = ['#wgq', '#barriers+food', '#access+food', 
                                                                '#barriers+cash', '#access+cash', '#personal'], only_these=True)
# make networkX graph
Gr_disability, pos_disability, weights_disability = mi_graph_disability.setup_nx_graph(E_disability, graph_scale = 0.4)
```

     ..... MI graph computed in 6.94 secs ...... 
     ..... NX graph computed in 0.02 secs ...... 



```python
mi_graph_disability.draw_graph(Gr_disability, pos_disability, weights_disability, link_weight=2., fig_size = (30,20))
```


![png](output_28_0.png)


Here are the top 20 pairs of variables based on the mutual information between them


```python
df = mi_graph_disability.save_top_mi_values(E_disability, topK=20)
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node1</th>
      <th>node1name</th>
      <th>node2</th>
      <th>node2name</th>
      <th>mutual_info</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>59</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>0.486406</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>60</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>0.383348</td>
    </tr>
    <tr>
      <th>2</th>
      <td>58</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>61</td>
      <td>#barriers+food_Services are delivered in place...</td>
      <td>0.349074</td>
    </tr>
    <tr>
      <th>3</th>
      <td>59</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>60</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>0.329437</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>#barriers+food_I do not have documents to acce...</td>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>0.293384</td>
    </tr>
    <tr>
      <th>5</th>
      <td>54</td>
      <td>#barriers+food_Safety fears for movement outsi...</td>
      <td>55</td>
      <td>#barriers+food_Safety fears for movement outsi...</td>
      <td>0.276852</td>
    </tr>
    <tr>
      <th>6</th>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>68</td>
      <td>#access+cash_Does your family receive any cash...</td>
      <td>0.253620</td>
    </tr>
    <tr>
      <th>7</th>
      <td>56</td>
      <td>#barriers+food_Services are not available?25</td>
      <td>68</td>
      <td>#access+cash_Does your family receive any cash...</td>
      <td>0.241075</td>
    </tr>
    <tr>
      <th>8</th>
      <td>56</td>
      <td>#barriers+food_Services are not available?25</td>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>0.239712</td>
    </tr>
    <tr>
      <th>9</th>
      <td>72</td>
      <td>#barriers+cash_Safety fears for movement outsi...</td>
      <td>73</td>
      <td>#barriers+cash_Safety fears for movement outsi...</td>
      <td>0.220503</td>
    </tr>
    <tr>
      <th>10</th>
      <td>60</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>61</td>
      <td>#barriers+food_Services are delivered in place...</td>
      <td>0.218725</td>
    </tr>
    <tr>
      <th>11</th>
      <td>59</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>61</td>
      <td>#barriers+food_Services are delivered in place...</td>
      <td>0.218725</td>
    </tr>
    <tr>
      <th>12</th>
      <td>53</td>
      <td>#barriers+food_I do not have documents to acce...</td>
      <td>68</td>
      <td>#access+cash_Does your family receive any cash...</td>
      <td>0.206535</td>
    </tr>
    <tr>
      <th>13</th>
      <td>62</td>
      <td>#barriers+food_Services are delivered in place...</td>
      <td>63</td>
      <td>#barriers+food_Staff are not supportive and/or...</td>
      <td>0.191314</td>
    </tr>
    <tr>
      <th>14</th>
      <td>57</td>
      <td>#barriers+food_Services are too expensive ?26</td>
      <td>58</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>0.187507</td>
    </tr>
    <tr>
      <th>15</th>
      <td>75</td>
      <td>#barriers+cash_Services are too expensive ?43</td>
      <td>77</td>
      <td>#barriers+cash_Services are far away and trans...</td>
      <td>0.161052</td>
    </tr>
    <tr>
      <th>16</th>
      <td>57</td>
      <td>#barriers+food_Services are too expensive ?26</td>
      <td>59</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>0.157185</td>
    </tr>
    <tr>
      <th>17</th>
      <td>76</td>
      <td>#barriers+cash_Services are far away and trans...</td>
      <td>78</td>
      <td>#barriers+cash_Services are far away and trans...</td>
      <td>0.148895</td>
    </tr>
    <tr>
      <th>18</th>
      <td>57</td>
      <td>#barriers+food_Services are too expensive ?26</td>
      <td>61</td>
      <td>#barriers+food_Services are delivered in place...</td>
      <td>0.136233</td>
    </tr>
    <tr>
      <th>19</th>
      <td>59</td>
      <td>#barriers+food_Services are far away and trans...</td>
      <td>68</td>
      <td>#access+cash_Does your family receive any cash...</td>
      <td>0.113399</td>
    </tr>
  </tbody>
</table>
</div>



How can we identify very "important" variables ? One way of looking at it is through the lens of Graph Theory. If a variable is highly correlated with several variables, which in turn are correlated with others, this variable has a higher value "centrality" in the graph. Let's compute "eigenvector" centrality : this is based on the graph adjacency matrix's principal eigenvector.


```python
C = mi_graph_disability.compute_centrality(Gr_disability, centrality_type='eigenvector')
```

the variable with the higest centrality is : 


```python
print(' .... Variable name : %s , centrality %.3f ....' %(data_disability.columns[C[0][0]], C[0][1]))
```

     .... Variable name : #barriers+food_Services are far away and transportation is not available?27 , centrality 0.515 ....



```python

```

# lets look at a graph that must include #disability as one of the nodes in each pair

We can re-use mi_graph because we only need to select a part of the graph!


```python
# now lets sub-select a part of the graph. We have to set only_these to False becuase that will look for #disability in both nodes in a pair
E = mi_graph.select_nodes(select_group = ['#disability'], only_these=False)
```

How many nodes are there in this sub-graph ? 


```python
len(E.keys())
```




    103




```python
Gr, pos, weights = mi_graph.setup_nx_graph(E, MI_thresh=0.005, graph_scale=0.5)
```

     ..... NX graph computed in 0.01 secs ...... 


now lets draw it !


```python
mi_graph.draw_graph(Gr, pos, weights, link_weight=2., fig_size = (30,20),
                    savepath ='')
```

     ..... NX graph plotted in 0.11 secs ..... 



![png](output_42_1.png)



```python
df = mi_graph.save_top_mi_values(E, topK=20, savepath='/Users/gagan/Dropbox/Work/UNOCHA/code/HI_Jordan/Results/HI_adult_disability_only_correlations.csv')
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node1</th>
      <th>node1name</th>
      <th>node2</th>
      <th>node2name</th>
      <th>mutual_info</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>93</td>
      <td>#wgq+ss_Do you have difficulty seeing, even wh...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.170313</td>
    </tr>
    <tr>
      <th>1</th>
      <td>97</td>
      <td>#wgq+ss_Do you have difficulty walking or clim...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.145440</td>
    </tr>
    <tr>
      <th>2</th>
      <td>114</td>
      <td>#wgq+en_How often do you feel depressed?</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.057466</td>
    </tr>
    <tr>
      <th>3</th>
      <td>111</td>
      <td>#wgq+en_Do you have difficulty using your hand...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.056287</td>
    </tr>
    <tr>
      <th>4</th>
      <td>108</td>
      <td>#wgq+ss_Do you have difficulty remembering or ...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.053552</td>
    </tr>
    <tr>
      <th>5</th>
      <td>109</td>
      <td>#wgq+ss_Do you have difficulty with self care,...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.051306</td>
    </tr>
    <tr>
      <th>6</th>
      <td>110</td>
      <td>#wgq+en_Do you have difficulty raising a 2 lit...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.049450</td>
    </tr>
    <tr>
      <th>7</th>
      <td>116</td>
      <td>#wgq+es_In the past 3 months, how often did yo...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.037593</td>
    </tr>
    <tr>
      <th>8</th>
      <td>106</td>
      <td>#wgq+ss_Using your usual language, do you have...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.023320</td>
    </tr>
    <tr>
      <th>9</th>
      <td>90</td>
      <td>#personal_Age</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.023165</td>
    </tr>
    <tr>
      <th>10</th>
      <td>112</td>
      <td>#wgq+en_How often do you feel worried, nervous...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.022954</td>
    </tr>
    <tr>
      <th>11</th>
      <td>117</td>
      <td>#wgq+es_Thinking about the last time you felt ...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.018086</td>
    </tr>
    <tr>
      <th>12</th>
      <td>73</td>
      <td>#barriers+cash_Safety fears for movement outsi...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.014007</td>
    </tr>
    <tr>
      <th>13</th>
      <td>26</td>
      <td>#barriers+medical_Among those mentioned as rea...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.010632</td>
    </tr>
    <tr>
      <th>14</th>
      <td>115</td>
      <td>#wgq+en_Thinking about the last time you felt ...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.010168</td>
    </tr>
    <tr>
      <th>15</th>
      <td>44</td>
      <td>#barriers+water_Among those mentioned as reaso...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.008778</td>
    </tr>
    <tr>
      <th>16</th>
      <td>67</td>
      <td>#barriers+food_Among those mentioned as reason...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.008578</td>
    </tr>
    <tr>
      <th>17</th>
      <td>23</td>
      <td>#barriers+medical_Services are not meeting my/...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.008226</td>
    </tr>
    <tr>
      <th>18</th>
      <td>87</td>
      <td>#access+cash+income_What is your household tot...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.007856</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>#barriers+medical_Services are far away and tr...</td>
      <td>119</td>
      <td>#disability</td>
      <td>0.006847</td>
    </tr>
  </tbody>
</table>
</div>



# save the networkX Graph Gr and the edges 


```python
mi_graph.save(Gr, '/Users/gagan/Dropbox/Work/UNOCHA/code/HI_Jordan/Results/HI_adult_mi_graph')
```


```python

```
