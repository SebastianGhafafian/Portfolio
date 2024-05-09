# Analysis of Data Science Job Postings

## Introduction
Starting out in the field of data science can be intimidating. There are alot of job postings online and the requirements vary drastically as data science is a broad field.
What better way to do create insights of the job market then to utilize the data science methodology itself?
This project cleans and create insights from data provided by Glassdoor.

## A first look at the data


```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv("Uncleaned_DS_jobs.csv")
df.head(8)

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
      <th>index</th>
      <th>Job Title</th>
      <th>Salary Estimate</th>
      <th>Job Description</th>
      <th>Rating</th>
      <th>Company Name</th>
      <th>Location</th>
      <th>Headquarters</th>
      <th>Size</th>
      <th>Founded</th>
      <th>Type of ownership</th>
      <th>Industry</th>
      <th>Sector</th>
      <th>Revenue</th>
      <th>Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Sr Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Description\n\nThe Senior Data Scientist is re...</td>
      <td>3.1</td>
      <td>Healthfirst\n3.1</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>1001 to 5000 employees</td>
      <td>1993</td>
      <td>Nonprofit Organization</td>
      <td>Insurance Carriers</td>
      <td>Insurance</td>
      <td>Unknown / Non-Applicable</td>
      <td>EmblemHealth, UnitedHealth Group, Aetna</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Secure our Nation, Ignite your Future\n\nJoin ...</td>
      <td>4.2</td>
      <td>ManTech\n4.2</td>
      <td>Chantilly, VA</td>
      <td>Herndon, VA</td>
      <td>5001 to 10000 employees</td>
      <td>1968</td>
      <td>Company - Public</td>
      <td>Research &amp; Development</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Overview\n\n\nAnalysis Group is one of the lar...</td>
      <td>3.8</td>
      <td>Analysis Group\n3.8</td>
      <td>Boston, MA</td>
      <td>Boston, MA</td>
      <td>1001 to 5000 employees</td>
      <td>1981</td>
      <td>Private Practice / Firm</td>
      <td>Consulting</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>JOB DESCRIPTION:\n\nDo you have a passion for ...</td>
      <td>3.5</td>
      <td>INFICON\n3.5</td>
      <td>Newton, MA</td>
      <td>Bad Ragaz, Switzerland</td>
      <td>501 to 1000 employees</td>
      <td>2000</td>
      <td>Company - Public</td>
      <td>Electrical &amp; Electronic Manufacturing</td>
      <td>Manufacturing</td>
      <td>$100 to $500 million (USD)</td>
      <td>MKS Instruments, Pfeiffer Vacuum, Agilent Tech...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Data Scientist\nAffinity Solutions / Marketing...</td>
      <td>2.9</td>
      <td>Affinity Solutions\n2.9</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>51 to 200 employees</td>
      <td>1998</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>Commerce Signals, Cardlytics, Yodlee</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>About Us:\n\nHeadquartered in beautiful Santa ...</td>
      <td>4.2</td>
      <td>HG Insights\n4.2</td>
      <td>Santa Barbara, CA</td>
      <td>Santa Barbara, CA</td>
      <td>51 to 200 employees</td>
      <td>2010</td>
      <td>Company - Private</td>
      <td>Computer Hardware &amp; Software</td>
      <td>Information Technology</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Data Scientist / Machine Learning Expert</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Posting Title\nData Scientist / Machine Learni...</td>
      <td>3.9</td>
      <td>Novartis\n3.9</td>
      <td>Cambridge, MA</td>
      <td>Basel, Switzerland</td>
      <td>10000+ employees</td>
      <td>1996</td>
      <td>Company - Public</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>$10+ billion (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Introduction\n\nHave you always wanted to run ...</td>
      <td>3.5</td>
      <td>iRobot\n3.5</td>
      <td>Bedford, MA</td>
      <td>Bedford, MA</td>
      <td>1001 to 5000 employees</td>
      <td>1990</td>
      <td>Company - Public</td>
      <td>Consumer Electronics &amp; Appliances Stores</td>
      <td>Retail</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>



There are immediatly a few points which are easily addressed:
1) The salary is not an integer and additional information is provided
2) The company name contains artefacts from the rating
3) The location can be split is state and city
4) The data contains inconsistent formats for unknown values ('unknown',-1,...)

Let's start with the salary:
### 1) Salary


```python
#Remove additional info for salary
df['Salary Estimate'] = df['Salary Estimate'].map(lambda x: x.split('(')[0])
df['Salary Estimate'].unique() #check result

```




    array(['$137K-$171K ', '$75K-$131K ', '$79K-$131K ', '$99K-$132K ',
           '$90K-$109K ', '$101K-$165K ', '$56K-$97K ', '$79K-$106K ',
           '$71K-$123K ', '$90K-$124K ', '$91K-$150K ', '$141K-$225K ',
           '$145K-$225K', '$79K-$147K ', '$122K-$146K ', '$112K-$116K ',
           '$110K-$163K ', '$124K-$198K ', '$79K-$133K ', '$69K-$116K ',
           '$31K-$56K ', '$95K-$119K ', '$212K-$331K ', '$66K-$112K ',
           '$128K-$201K ', '$138K-$158K ', '$80K-$132K ', '$87K-$141K ',
           '$92K-$155K ', '$105K-$167K '], dtype=object)



All that's left is to extract the numerical values for the minimum, maximum and mean:


```python
# Function to extract minimum and maximum salary values
def extract_salary(salary_estimate):
    # Split the minimum from maximum salary
    min_salary, max_salary = salary_estimate.split('-')
    
    # Remove the '$' and 'K' characters and convert to integers
    min_salary = int(min_salary.strip('$').replace('K', '')) * 1000
    max_salary = int(max_salary.strip('$').replace('K', '')) * 1000
    
    return min_salary, max_salary

# Apply the function to each element in the 'Salary Estimate' column
df['Salary Minimum'], df['Salary Maximum'] = zip(*df['Salary Estimate'].apply(extract_salary))

# Calculate the average salary
df['Salary Average'] = (df['Salary Minimum'] + df['Salary Maximum']) / 2

# Using the values from the min_salary and max_salary to put into the Salary Estimate column
df.drop(columns='Salary Estimate', inplace=True)
```

### 2) Company name


```python
df['Company Name'] = df['Company Name'].map(lambda x: x.split('\n')[0])
```

### 3) Location


```python
df['State'] = df['Location'].apply(
    lambda x: x.split(',')[1][1:] if ',' in x else pd.NA)
df['State'].value_counts()
df['City'] = df['Location'].apply(
    lambda x: x.split(',')[0] if ',' in x else pd.NA)
```

### 4) Fix inconsistent formats


```python
fix_cols = ['Job Title', 'Job Description', 'Company Name', 'Location', 'Headquarters', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Competitors']

df[fix_cols] = df[fix_cols].replace('-1', pd.NA)
df["Rating"] = df["Rating"].replace(-1, pd.NA)
df[fix_cols] = df[fix_cols].replace('Unknown', pd.NA)
df['Revenue'] = np.where(df['Revenue']=="Unknown / Non-Applicable",pd.NA,df['Revenue'])
df['Revenue'].value_counts(dropna=False)


```




    Revenue
    <NA>                                240
    $100 to $500 million (USD)           94
    $10+ billion (USD)                   63
    $2 to $5 billion (USD)               45
    $10 to $25 million (USD)             41
    $1 to $2 billion (USD)               36
    $25 to $50 million (USD)             36
    $50 to $100 million (USD)            31
    $1 to $5 million (USD)               31
    $500 million to $1 billion (USD)     19
    $5 to $10 million (USD)              14
    Less than $1 million (USD)           14
    $5 to $10 billion (USD)               8
    Name: count, dtype: int64




```python

```


```python
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
      <th>index</th>
      <th>Job Title</th>
      <th>Job Description</th>
      <th>Rating</th>
      <th>Company Name</th>
      <th>Location</th>
      <th>Headquarters</th>
      <th>Size</th>
      <th>Founded</th>
      <th>Type of ownership</th>
      <th>Industry</th>
      <th>Sector</th>
      <th>Revenue</th>
      <th>Competitors</th>
      <th>Salary Minimum</th>
      <th>Salary Maximum</th>
      <th>Salary Average</th>
      <th>State</th>
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Sr Data Scientist</td>
      <td>Description\n\nThe Senior Data Scientist is re...</td>
      <td>3.1</td>
      <td>Healthfirst</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>1001 to 5000 employees</td>
      <td>1993</td>
      <td>Nonprofit Organization</td>
      <td>Insurance Carriers</td>
      <td>Insurance</td>
      <td>&lt;NA&gt;</td>
      <td>EmblemHealth, UnitedHealth Group, Aetna</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>NY</td>
      <td>New York</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Data Scientist</td>
      <td>Secure our Nation, Ignite your Future\n\nJoin ...</td>
      <td>4.2</td>
      <td>ManTech</td>
      <td>Chantilly, VA</td>
      <td>Herndon, VA</td>
      <td>5001 to 10000 employees</td>
      <td>1968</td>
      <td>Company - Public</td>
      <td>Research &amp; Development</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>&lt;NA&gt;</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>VA</td>
      <td>Chantilly</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Data Scientist</td>
      <td>Overview\n\n\nAnalysis Group is one of the lar...</td>
      <td>3.8</td>
      <td>Analysis Group</td>
      <td>Boston, MA</td>
      <td>Boston, MA</td>
      <td>1001 to 5000 employees</td>
      <td>1981</td>
      <td>Private Practice / Firm</td>
      <td>Consulting</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>&lt;NA&gt;</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>MA</td>
      <td>Boston</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Data Scientist</td>
      <td>JOB DESCRIPTION:\n\nDo you have a passion for ...</td>
      <td>3.5</td>
      <td>INFICON</td>
      <td>Newton, MA</td>
      <td>Bad Ragaz, Switzerland</td>
      <td>501 to 1000 employees</td>
      <td>2000</td>
      <td>Company - Public</td>
      <td>Electrical &amp; Electronic Manufacturing</td>
      <td>Manufacturing</td>
      <td>$100 to $500 million (USD)</td>
      <td>MKS Instruments, Pfeiffer Vacuum, Agilent Tech...</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>MA</td>
      <td>Newton</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Data Scientist</td>
      <td>Data Scientist\nAffinity Solutions / Marketing...</td>
      <td>2.9</td>
      <td>Affinity Solutions</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>51 to 200 employees</td>
      <td>1998</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>&lt;NA&gt;</td>
      <td>Commerce Signals, Cardlytics, Yodlee</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>NY</td>
      <td>New York</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Data Scientist</td>
      <td>About Us:\n\nHeadquartered in beautiful Santa ...</td>
      <td>4.2</td>
      <td>HG Insights</td>
      <td>Santa Barbara, CA</td>
      <td>Santa Barbara, CA</td>
      <td>51 to 200 employees</td>
      <td>2010</td>
      <td>Company - Private</td>
      <td>Computer Hardware &amp; Software</td>
      <td>Information Technology</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>CA</td>
      <td>Santa Barbara</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Data Scientist / Machine Learning Expert</td>
      <td>Posting Title\nData Scientist / Machine Learni...</td>
      <td>3.9</td>
      <td>Novartis</td>
      <td>Cambridge, MA</td>
      <td>Basel, Switzerland</td>
      <td>10000+ employees</td>
      <td>1996</td>
      <td>Company - Public</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>$10+ billion (USD)</td>
      <td>&lt;NA&gt;</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>MA</td>
      <td>Cambridge</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>Data Scientist</td>
      <td>Introduction\n\nHave you always wanted to run ...</td>
      <td>3.5</td>
      <td>iRobot</td>
      <td>Bedford, MA</td>
      <td>Bedford, MA</td>
      <td>1001 to 5000 employees</td>
      <td>1990</td>
      <td>Company - Public</td>
      <td>Consumer Electronics &amp; Appliances Stores</td>
      <td>Retail</td>
      <td>$1 to $2 billion (USD)</td>
      <td>&lt;NA&gt;</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>MA</td>
      <td>Bedford</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>Staff Data Scientist - Analytics</td>
      <td>Intuit is seeking a Staff Data Scientist to co...</td>
      <td>4.4</td>
      <td>Intuit - Data</td>
      <td>San Diego, CA</td>
      <td>Mountain View, CA</td>
      <td>5001 to 10000 employees</td>
      <td>1983</td>
      <td>Company - Public</td>
      <td>Computer Hardware &amp; Software</td>
      <td>Information Technology</td>
      <td>$2 to $5 billion (USD)</td>
      <td>Square, PayPal, H&amp;R Block</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>CA</td>
      <td>San Diego</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>Data Scientist</td>
      <td>Ready to write the best chapter of your career...</td>
      <td>3.6</td>
      <td>XSELL Technologies</td>
      <td>Chicago, IL</td>
      <td>Chicago, IL</td>
      <td>51 to 200 employees</td>
      <td>2014</td>
      <td>Company - Private</td>
      <td>Enterprise Software &amp; Network Solutions</td>
      <td>Information Technology</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>IL</td>
      <td>Chicago</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>Data Scientist</td>
      <td>Join our team dedicated to developing and exec...</td>
      <td>4.5</td>
      <td>Novetta</td>
      <td>Herndon, VA</td>
      <td>Mc Lean, VA</td>
      <td>501 to 1000 employees</td>
      <td>2012</td>
      <td>Company - Private</td>
      <td>Enterprise Software &amp; Network Solutions</td>
      <td>Information Technology</td>
      <td>$100 to $500 million (USD)</td>
      <td>Leidos, CACI International, Booz Allen Hamilton</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>VA</td>
      <td>Herndon</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>Data Scientist</td>
      <td>About Us\n\n\nInterested in working for a huma...</td>
      <td>4.7</td>
      <td>1904labs</td>
      <td>Saint Louis, MO</td>
      <td>Saint Louis, MO</td>
      <td>51 to 200 employees</td>
      <td>2016</td>
      <td>Company - Private</td>
      <td>IT Services</td>
      <td>Information Technology</td>
      <td>&lt;NA&gt;</td>
      <td>Slalom, Daugherty Business Solutions</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>MO</td>
      <td>Saint Louis</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>Data Scientist - Statistics, Early Career</td>
      <td>*Organization and Job ID**\nJob ID: 310918\n\n...</td>
      <td>3.7</td>
      <td>PNNL</td>
      <td>Richland, WA</td>
      <td>Richland, WA</td>
      <td>1001 to 5000 employees</td>
      <td>1965</td>
      <td>Government</td>
      <td>Energy</td>
      <td>Oil, Gas, Energy &amp; Utilities</td>
      <td>$500 million to $1 billion (USD)</td>
      <td>Oak Ridge National Laboratory, National Renewa...</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>WA</td>
      <td>Richland</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>Data Modeler</td>
      <td>POSITION PURPOSE:\n\nThe Data Architect/Data M...</td>
      <td>3.1</td>
      <td>Old World Industries</td>
      <td>Northbrook, IL</td>
      <td>Northbrook, IL</td>
      <td>201 to 500 employees</td>
      <td>1973</td>
      <td>Company - Private</td>
      <td>Chemical Manufacturing</td>
      <td>Manufacturing</td>
      <td>$1 to $2 billion (USD)</td>
      <td>&lt;NA&gt;</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>IL</td>
      <td>Northbrook</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>Data Scientist</td>
      <td>Position Description:\n\nWant to make a differ...</td>
      <td>3.4</td>
      <td>Mathematica Policy Research</td>
      <td>Washington, DC</td>
      <td>Princeton, NJ</td>
      <td>1001 to 5000 employees</td>
      <td>1986</td>
      <td>Company - Private</td>
      <td>Consulting</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>&lt;NA&gt;</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>DC</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>Experienced Data Scientist</td>
      <td>*******Please Apply using this link: https://a...</td>
      <td>4.4</td>
      <td>Guzman &amp; Griffin Technologies (GGTI)</td>
      <td>Washington, DC</td>
      <td>Mays Landing, NJ</td>
      <td>1 to 50 employees</td>
      <td>1997</td>
      <td>Company - Private</td>
      <td>Federal Agencies</td>
      <td>Government</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>DC</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>Data Scientist - Contract</td>
      <td>We are an ambitious, well-funded startup with ...</td>
      <td>4.1</td>
      <td>Upside Business Travel</td>
      <td>Remote</td>
      <td>Washington, DC</td>
      <td>51 to 200 employees</td>
      <td>2015</td>
      <td>Company - Private</td>
      <td>Internet</td>
      <td>Information Technology</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>Data Scientist</td>
      <td>Job Success Profile\n\nData Scientist\n\nBuckm...</td>
      <td>3.5</td>
      <td>Buckman</td>
      <td>Memphis, TN</td>
      <td>Memphis, TN</td>
      <td>1001 to 5000 employees</td>
      <td>1945</td>
      <td>Company - Private</td>
      <td>Chemical Manufacturing</td>
      <td>Manufacturing</td>
      <td>$500 million to $1 billion (USD)</td>
      <td>&lt;NA&gt;</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>TN</td>
      <td>Memphis</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>Data Analyst II</td>
      <td>The Data Analyst II is responsible for data en...</td>
      <td>4.2</td>
      <td>Insight Enterprises, Inc.</td>
      <td>Plano, TX</td>
      <td>Tempe, AZ</td>
      <td>5001 to 10000 employees</td>
      <td>1988</td>
      <td>Company - Public</td>
      <td>Enterprise Software &amp; Network Solutions</td>
      <td>Information Technology</td>
      <td>$5 to $10 billion (USD)</td>
      <td>CDW, PCM, SHI International</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>TX</td>
      <td>Plano</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>Medical Lab Scientist</td>
      <td>Responsibilities\n\n\nThe Medical Laboratory S...</td>
      <td>3.5</td>
      <td>Tower Health</td>
      <td>West Grove, PA</td>
      <td>Reading, PA</td>
      <td>5001 to 10000 employees</td>
      <td>2017</td>
      <td>Nonprofit Organization</td>
      <td>Health Care Services &amp; Hospitals</td>
      <td>Health Care</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>137000</td>
      <td>171000</td>
      <td>154000.0</td>
      <td>PA</td>
      <td>West Grove</td>
    </tr>
  </tbody>
</table>
</div>



Let's turn the more challenging features:
1) Job titles are not uniform
2) Job titles may contain information about the level ('Sr', 'Junior') and could be extracted as features
2) Job requirements are "hidden" in unstructured feature "Job Description"


```python
pd.set_option('display.max_rows', 10)
df["Job Title"].value_counts()
```




    Job Title
    Data Scientist                                            337
    Data Engineer                                              26
    Senior Data Scientist                                      19
    Machine Learning Engineer                                  16
    Data Analyst                                               12
                                                             ... 
    Data Science Instructor                                     1
    Business Data Analyst                                       1
    Purification Scientist                                      1
    Data Engineer, Enterprise Analytics                         1
    AI/ML - Machine Learning Scientist, Siri Understanding      1
    Name: count, Length: 172, dtype: int64



Writing a classifier function for title helps to structure the Job titles partly.


```python
#creates simplified job titles
def title_classifier(title):

    if 'data scientist' in title.lower(): 
        return 'Data scientist'
    elif 'data engineer' in title.lower():
        return 'Data engineer'
    elif 'analyst' in title.lower():
        return 'Analyst'
    elif 'machine learning' in title.lower():
        return 'MLE'
    elif 'manager' in title.lower():
        return 'Manager'
    elif 'director' in title.lower():
        return 'Director'
    elif 'deep learning' in title.lower():
        return 'MLE'
    else:
        return 'Others'

```

Let's apply the classifier and check the results:


```python
df['Job Title simplyfied'] = df['Job Title'].apply(title_classifier)
df["Job Title simplyfied"].value_counts()
```




    Job Title simplyfied
    Data scientist    455
    Others             67
    Analyst            55
    Data engineer      47
    MLE                38
    Manager             7
    Director            3
    Name: count, dtype: int64



Most job titles were simplyfied. 67 entries remain unclassified. We can check the remaining values to verify the vast variety of the remaining job titles


```python
pd.set_option('display.max_rows', 67)
df[df['Job Title simplyfied']=='NA']["Job Title"]
```




    Series([], Name: Job Title, dtype: object)



### 2) Job levels


```python
#extracts level from  job title
def level_classifier(title):
    if 'jr' in title.lower() or 'junior' in title.lower():
        return 'Junior'
    if 'sr' in title.lower() or 'senior' in title.lower():
        return 'Senior'
    else: return 'NA'
```


```python
df['Level'] = df['Job Title'].apply(level_classifier)
df['Level'].value_counts()
```




    Level
    NA        594
    Senior     76
    Junior      2
    Name: count, dtype: int64



We can see that most positions don't specify the level and only 2 Junior positions are offered of the total 672 entries!

### 3) Job requirements
The job descriptions carries the most information, yet the information is highly unstructured. Keywords like Python, R and SQL are easily detected and Indicator variables can be created. Yet, on of the most interesting information namely the number of years of experience remain uncovered as uncovering it is very challenging due to the data structure and potentially multiple statements in dependancy of the education level and other factors.


```python
df['python']   = df['Job Description'].map(lambda x: 1 if 'python'   in x.lower() else 0)
df['excel']    = df['Job Description'].map(lambda x: 1 if 'excel'    in x.lower() else 0)
df['hadoop']   = df['Job Description'].map(lambda x: 1 if 'hadoop'   in x.lower() else 0)
df['spark']    = df['Job Description'].map(lambda x: 1 if 'spark'    in x.lower() else 0)
df['aws']      = df['Job Description'].map(lambda x: 1 if 'aws'      in x.lower() else 0)
df['tableau']  = df['Job Description'].map(lambda x: 1 if 'tableau'  in x.lower() else 0)
df['big_data'] = df['Job Description'].map(lambda x: 1 if 'big data' in x.lower() else 0)
df['sql']      = df['Job Description'].map(lambda x: 1 if 'sql'      in x.lower() else 0)
df['r']        = df['Job Description'].map(lambda x: 1 if ' R '      in x or ' R.' in x else 0)

```

This concludes the data cleaning part. The data frame is safed as a csv file. 


```python
df.to_csv('cleaned_jobs.csv')
```

## Exploratory Data Analysis

Insights to be gained:
* How many senior/ junior / unspec roles
* Does location affect salary
* Does competition drive salary
* (Scraping) Which state offers the most positions and does it reflect the population
* (Scraping) Closer look on CA: How is the distribution there.
* What are the companies offering jobs in LA
* x What are the most desired job requirements
* x What are the most common job titles
* Does the programming language affect salary
* What field uses most python, R, sql?
* Is company rating influence by salary
* Create a word map for the job description




```python
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('cleaned_jobs.csv')
df.head()
df.columns

```




    Index(['Unnamed: 0', 'index', 'Job Title', 'Job Description', 'Rating',
           'Company Name', 'Location', 'Headquarters', 'Size', 'Founded',
           'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Competitors',
           'Salary Minimum', 'Salary Maximum', 'Salary Average', 'State', 'City',
           'Job Title simplyfied', 'Level', 'python', 'excel', 'hadoop', 'spark',
           'aws', 'tableau', 'big_data', 'sql', 'r'],
          dtype='object')




```python
sns.set_theme()
requirements = ['python','sql', 'excel', 'hadoop', 'spark',
       'aws', 'tableau', 'big_data',  'r']
_ = sns.barplot(x= df["Job Title simplyfied"].value_counts().index,
            y=df["Job Title simplyfied"].value_counts()/df.shape[0]*100,
            hue = df["Job Title simplyfied"].value_counts().index)
plt.title('Overview over different job titles')
plt.ylabel('Percentage of Job offers')
plt.xlabel('Job title')
plt.xticks(rotation=45);
```

    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)



    
![png](EDA_files/EDA_35_1.png)
    



```python
df[requirements].mean().shape

```




    (9,)




```python

sns.barplot(x=requirements,
            y=df[requirements].mean(),
            hue = df[requirements].mean().index)
plt.title('Relative Amount of Jobs posting asking for specific requirement')
plt.ylabel('Relative Amount in Percent')
plt.xlabel('Requirement')
plt.xticks(rotation=45);

```

    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)



    
![png](EDA_files/EDA_37_1.png)
    



```python
fig = plt.subplots(figsize = (15,8))
_ = sns.barplot(x= df["State"].value_counts().index,
            y=df["State"].value_counts()/df.shape[0]*100,
            hue = df["State"].value_counts().index)
plt.title('Overview over different job titles')
plt.ylabel('Percentage of Job offers')
plt.xlabel('Job title')
plt.xticks(rotation=45);
```

    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)



    
![png](EDA_files/EDA_38_1.png)
    



```python
df_ca = df[df['State'] == 'CA']

fig = plt.subplots(figsize = (15,8))
ax = sns.barplot(x= df_ca["City"].value_counts().index,
            y=df_ca["City"].value_counts(),
            hue = df_ca["City"].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.title('Offered positions in California')
plt.ylabel('Percentage of Job offers')
plt.xlabel('City')
plt.xticks(rotation=90);
```

    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/_base.py:949: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    /tmp/ipykernel_44336/1501044289.py:7: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
      ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)



    
![png](EDA_files/EDA_39_1.png)
    



```python
df_ca["City"].value_counts()
```




    City
    San Francisco          69
    Santa Clara             9
    Redwood City            7
    San Diego               7
    Palo Alto               5
    Thousand Oaks           5
    San Jose                4
    San Carlos              4
    South San Francisco     4
    Sunnyvale               3
    Carson                  3
    Cupertino               3
    Santa Barbara           3
    Scotts Valley           2
    Concord                 2
    Irvine                  2
    San Clemente            2
    Oakland                 2
    Livermore               2
    Santa Cruz              2
    Emeryville              2
    Santa Monica            2
    San Mateo               2
    Carpinteria             1
    Oxnard                  1
    Oakville                1
    Orange                  1
    Rancho Cucamonga        1
    Simi Valley             1
    Pleasanton              1
    Foster City             1
    Mountain View           1
    Burbank                 1
    San Ramon               1
    Fremont                 1
    Brisbane                1
    Monterey                1
    Sacramento              1
    Menlo Park              1
    Burlingame              1
    Culver City             1
    Irwindale               1
    Name: count, dtype: int64




```python

import requests
from bs4 import BeautifulSoup
import json  

link = requests.get("https://en.wikipedia.org/wiki/List_of_districts_and_neighborhoods_in_Los_Angeles")
soup = BeautifulSoup(link.text, "lxml")


```


```python
places = soup.find_all('ul')
len(places)
```




    51




```python
sections=soup.find_all(class_="div-col columns column-width")
places = BeautifulSoup(str(sections)).find_all('li')

neighborhoods_list = []
places
```




    []




```python

for div in places:
    if div.find('a').contents[0] == '[40]':
        neighborhoods_list.append('Pico Robertson')
    else:
        neighborhoods_list.append(div.find('a').contents[0])

neighborhoods_list
```




    []



## Salary differences across states



```python

```


```python

```


```python
states = df.groupby("State").mean(numeric_only=True)

fig = plt.subplots(figsize = (15,8))
sns.boxplot(data=df, x = "State", y = "Salary Average")

```

    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/categorical.py:640: FutureWarning: SeriesGroupBy.grouper is deprecated and will be removed in a future version of pandas.
      positions = grouped.grouper.result_index.to_numpy(dtype=float)





    <Axes: xlabel='State', ylabel='Salary Average'>




    
![png](EDA_files/EDA_48_2.png)
    


Wisconsin and IA, NC why so high median, what kind of industry?
Tie together with industry averages


```python
fig = plt.subplots(figsize = (15,8))
sns.lineplot( x = states.index, y = states['Salary Maximum'],drawstyle='steps')
sns.lineplot( x = states.index, y = states['Salary Minimum'],drawstyle='steps-pre')
sns.lineplot( x = states.index, y = states['Salary Average'],drawstyle='steps-pre')
plt.xticks(rotation=90);
```


    
![png](EDA_files/EDA_50_0.png)
    



```python
df[df["State"]== "NY"][["Job Title", "Salary Average", "Salary Minimum", "Salary Maximum"]]
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
      <th>Job Title</th>
      <th>Salary Average</th>
      <th>Salary Minimum</th>
      <th>Salary Maximum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sr Data Scientist</td>
      <td>154000.0</td>
      <td>137000</td>
      <td>171000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Data Scientist</td>
      <td>154000.0</td>
      <td>137000</td>
      <td>171000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Data Scientist</td>
      <td>154000.0</td>
      <td>137000</td>
      <td>171000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Data Scientist/Machine Learning</td>
      <td>154000.0</td>
      <td>137000</td>
      <td>171000</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Data Scientist</td>
      <td>105000.0</td>
      <td>79000</td>
      <td>131000</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Data Engineer, Enterprise Analytics</td>
      <td>105000.0</td>
      <td>79000</td>
      <td>131000</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Data Scientist</td>
      <td>105000.0</td>
      <td>79000</td>
      <td>131000</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Data Scientist, Kinship - NYC/Portland</td>
      <td>115500.0</td>
      <td>99000</td>
      <td>132000</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Data Scientist</td>
      <td>115500.0</td>
      <td>99000</td>
      <td>132000</td>
    </tr>
    <tr>
      <th>154</th>
      <td>ELISA RESEARCH SCIENTIST (CV-15)</td>
      <td>99500.0</td>
      <td>90000</td>
      <td>109000</td>
    </tr>
    <tr>
      <th>173</th>
      <td>Senior Data Scientist - Algorithms</td>
      <td>133000.0</td>
      <td>101000</td>
      <td>165000</td>
    </tr>
    <tr>
      <th>182</th>
      <td>Sr. Data Scientist II</td>
      <td>76500.0</td>
      <td>56000</td>
      <td>97000</td>
    </tr>
    <tr>
      <th>194</th>
      <td>Senior Data Scientist - R&amp;D Oncology</td>
      <td>76500.0</td>
      <td>56000</td>
      <td>97000</td>
    </tr>
    <tr>
      <th>221</th>
      <td>Data Scientist/Machine Learning</td>
      <td>97000.0</td>
      <td>71000</td>
      <td>123000</td>
    </tr>
    <tr>
      <th>234</th>
      <td>Sr. Data Scientist II</td>
      <td>97000.0</td>
      <td>71000</td>
      <td>123000</td>
    </tr>
    <tr>
      <th>242</th>
      <td>Software Data Engineer</td>
      <td>107000.0</td>
      <td>90000</td>
      <td>124000</td>
    </tr>
    <tr>
      <th>253</th>
      <td>Data Scientist</td>
      <td>107000.0</td>
      <td>90000</td>
      <td>124000</td>
    </tr>
    <tr>
      <th>254</th>
      <td>Data Scientist</td>
      <td>107000.0</td>
      <td>90000</td>
      <td>124000</td>
    </tr>
    <tr>
      <th>258</th>
      <td>Data Scientist</td>
      <td>107000.0</td>
      <td>90000</td>
      <td>124000</td>
    </tr>
    <tr>
      <th>269</th>
      <td>Data Engineer, Digital &amp; Comp Pathology</td>
      <td>120500.0</td>
      <td>91000</td>
      <td>150000</td>
    </tr>
    <tr>
      <th>272</th>
      <td>Data Scientist</td>
      <td>120500.0</td>
      <td>91000</td>
      <td>150000</td>
    </tr>
    <tr>
      <th>286</th>
      <td>Senior Data Scientist - R&amp;D Oncology</td>
      <td>183000.0</td>
      <td>141000</td>
      <td>225000</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Data Scientist</td>
      <td>183000.0</td>
      <td>141000</td>
      <td>225000</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Data Scientist</td>
      <td>183000.0</td>
      <td>141000</td>
      <td>225000</td>
    </tr>
    <tr>
      <th>301</th>
      <td>Data Scientist</td>
      <td>183000.0</td>
      <td>141000</td>
      <td>225000</td>
    </tr>
    <tr>
      <th>308</th>
      <td>Data Scientist</td>
      <td>185000.0</td>
      <td>145000</td>
      <td>225000</td>
    </tr>
    <tr>
      <th>310</th>
      <td>Data Scientist</td>
      <td>185000.0</td>
      <td>145000</td>
      <td>225000</td>
    </tr>
    <tr>
      <th>317</th>
      <td>Data Scientist</td>
      <td>185000.0</td>
      <td>145000</td>
      <td>225000</td>
    </tr>
    <tr>
      <th>319</th>
      <td>Data Scientist</td>
      <td>185000.0</td>
      <td>145000</td>
      <td>225000</td>
    </tr>
    <tr>
      <th>332</th>
      <td>Machine Learning Engineer/Scientist</td>
      <td>113000.0</td>
      <td>79000</td>
      <td>147000</td>
    </tr>
    <tr>
      <th>341</th>
      <td>Data Scientist</td>
      <td>113000.0</td>
      <td>79000</td>
      <td>147000</td>
    </tr>
    <tr>
      <th>342</th>
      <td>VP, Data Science</td>
      <td>113000.0</td>
      <td>79000</td>
      <td>147000</td>
    </tr>
    <tr>
      <th>349</th>
      <td>Data Scientist</td>
      <td>134000.0</td>
      <td>122000</td>
      <td>146000</td>
    </tr>
    <tr>
      <th>424</th>
      <td>Data Scientist</td>
      <td>161000.0</td>
      <td>124000</td>
      <td>198000</td>
    </tr>
    <tr>
      <th>425</th>
      <td>Data Scientist</td>
      <td>161000.0</td>
      <td>124000</td>
      <td>198000</td>
    </tr>
    <tr>
      <th>427</th>
      <td>Data Scientist</td>
      <td>106000.0</td>
      <td>79000</td>
      <td>133000</td>
    </tr>
    <tr>
      <th>445</th>
      <td>Machine Learning Engineer/Scientist</td>
      <td>106000.0</td>
      <td>79000</td>
      <td>133000</td>
    </tr>
    <tr>
      <th>460</th>
      <td>Data Scientist</td>
      <td>92500.0</td>
      <td>69000</td>
      <td>116000</td>
    </tr>
    <tr>
      <th>465</th>
      <td>Data Scientist</td>
      <td>92500.0</td>
      <td>69000</td>
      <td>116000</td>
    </tr>
    <tr>
      <th>469</th>
      <td>VP, Data Science</td>
      <td>43500.0</td>
      <td>31000</td>
      <td>56000</td>
    </tr>
    <tr>
      <th>490</th>
      <td>Data Scientist</td>
      <td>107000.0</td>
      <td>95000</td>
      <td>119000</td>
    </tr>
    <tr>
      <th>511</th>
      <td>Data Scientist(s)/Machine Learning Engineer</td>
      <td>271500.0</td>
      <td>212000</td>
      <td>331000</td>
    </tr>
    <tr>
      <th>524</th>
      <td>Data Scientist</td>
      <td>271500.0</td>
      <td>212000</td>
      <td>331000</td>
    </tr>
    <tr>
      <th>527</th>
      <td>Data Scientist</td>
      <td>271500.0</td>
      <td>212000</td>
      <td>331000</td>
    </tr>
    <tr>
      <th>546</th>
      <td>Data Scientist</td>
      <td>89000.0</td>
      <td>66000</td>
      <td>112000</td>
    </tr>
    <tr>
      <th>558</th>
      <td>Clinical Data Analyst</td>
      <td>164500.0</td>
      <td>128000</td>
      <td>201000</td>
    </tr>
    <tr>
      <th>567</th>
      <td>Machine Learning Engineer</td>
      <td>164500.0</td>
      <td>128000</td>
      <td>201000</td>
    </tr>
    <tr>
      <th>588</th>
      <td>Senior Data Engineer</td>
      <td>148000.0</td>
      <td>138000</td>
      <td>158000</td>
    </tr>
    <tr>
      <th>605</th>
      <td>Data Scientist</td>
      <td>106000.0</td>
      <td>80000</td>
      <td>132000</td>
    </tr>
    <tr>
      <th>631</th>
      <td>Data Scientist, Kinship - NYC/Portland</td>
      <td>123500.0</td>
      <td>92000</td>
      <td>155000</td>
    </tr>
    <tr>
      <th>640</th>
      <td>Data Scientist(s)/Machine Learning Engineer</td>
      <td>123500.0</td>
      <td>92000</td>
      <td>155000</td>
    </tr>
    <tr>
      <th>671</th>
      <td>Data Scientist</td>
      <td>136000.0</td>
      <td>105000</td>
      <td>167000</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.pivot_table(df, index='Job Title simplyfied', values='Salary Average')
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
      <th>Salary Average</th>
    </tr>
    <tr>
      <th>Job Title simplyfied</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Analyst</th>
      <td>115718.181818</td>
    </tr>
    <tr>
      <th>Data engineer</th>
      <td>113808.510638</td>
    </tr>
    <tr>
      <th>Data scientist</th>
      <td>125216.483516</td>
    </tr>
    <tr>
      <th>Director</th>
      <td>127333.333333</td>
    </tr>
    <tr>
      <th>MLE</th>
      <td>118986.842105</td>
    </tr>
    <tr>
      <th>Manager</th>
      <td>138214.285714</td>
    </tr>
    <tr>
      <th>Others</th>
      <td>127522.388060</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS


stop_words = list(STOPWORDS)
custom_stop_words = ['experience','user', 'need']
stop_words = set(stop_words + custom_stop_words)
words = " ".join(df['Job Description'])
wordcloud = WordCloud(max_words=400, width =1280, height = 720, background_color="black",stopwords=stop_words).generate(words)
plt.figure(figsize=[15,15])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
plt.savefig('wordcloud.png')
```


    
![png](EDA_files/EDA_54_0.png)
    



    <Figure size 640x480 with 0 Axes>



```python
wc = WordCloud(background_color='white', colormap = 'binary',
     stopwords = ['meta'], width = 800, height = 500).generate(words)
plt.axis("off")
plt.imshow(wc)
```




    <matplotlib.image.AxesImage at 0x739a12680130>




    
![png](EDA_files/EDA_55_1.png)
    



```python
words = " ".join(df['Job Description'])
wordcloud = WordCloud(max_words=5000, width =1280, height = 720, background_color="black").generate(words[0:100])
plt.figure(figsize=[15,15])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
plt.savefig('wordcloud.png')
```


    
![png](EDA_files/EDA_56_0.png)
    



    <Figure size 640x480 with 0 Axes>



```python

df_language = pd.DataFrame({'Python':[],'Python_R':[],'R':[]})
df_python   = df[(df['python']==1) & (df['r']==0)].assign(Location=1)
df_python_R = df[(df['python']==1) & (df['r']==1)].assign(Location=2)
df_R        = df[(df['python']==0) & (df['r']==1)].assign(Location=3)

df_R
cdf = pd.concat([df_python, df_python_R, df_R])    
mdf = pd.melt(cdf, id_vars=['Location'])
mdf
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
      <th>Location</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Unnamed: 0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Unnamed: 0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Unnamed: 0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Unnamed: 0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Unnamed: 0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15085</th>
      <td>3</td>
      <td>r</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15086</th>
      <td>3</td>
      <td>r</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15087</th>
      <td>3</td>
      <td>r</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15088</th>
      <td>3</td>
      <td>r</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15089</th>
      <td>3</td>
      <td>r</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>15090 rows × 3 columns</p>
</div>




```python
def get_language(python, r):
    if python == 1 and r == 1:
        return 'Python + R'
    elif python == 1 and r == 0:
        return 'Python'
    elif python == 0 and r == 1:
        return 'R'
    
```


```python
df['Language'] = df.apply(lambda x: get_language(python=x['python'],r=x['r']), axis=1)
ax = sns.boxplot(data=df, x='Language', y='Salary Average')
```

    /home/sebastian/.local/lib/python3.10/site-packages/seaborn/categorical.py:640: FutureWarning: SeriesGroupBy.grouper is deprecated and will be removed in a future version of pandas.
      positions = grouped.grouper.result_index.to_numpy(dtype=float)



    
![png](EDA_files/EDA_59_1.png)
    



```python
df_language

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
      <th>Python</th>
      <th>Python_R</th>
      <th>R</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


