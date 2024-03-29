Libraries:
ScalNet - https://github.com/eclipse/deeplearning4j/tree/master/scalnet?fbclid=IwAR01enpe_dySCpU1aPkMorznm6k31cDmQ49wE52_jAGQzcr-3CZs9NNSVas
Examples - https://github.com/deeplearning4j/ScalNet/tree/master/src/test/scala/org/deeplearning4j/scalnet/examples?fbclid=IwAR2uMjTESm9KHAIZ_mZCHckZhRuZJByhmAbQDoUAn1vCVC1SoE0KmKDmQ9M

ScalaNLP, Vegas, and Breeze:
https://github.com/scalanlp/breeze/
Breeze examples - https://github.com/scalanlp/breeze-examples
Breeze quickstart - https://github.com/scalanlp/breeze/wiki/Quickstart
Vegas - https://github.com/vegas-viz/Vegas


# Awesome Machine Learning

Конференции

Конференция NIPS (nips.cc)
Конференция Icml (icml.cc)

https://github.com/Yorko/mlcourse.ai/blob/master/jupyter_russian/topic01_pandas_data_analysis/lesson1_practice_pandas_titanic.ipynb

# Other
http://datareview.info/article/universalnyj-podxod-pochti-k-lyuboj-zadache-mashinnogo-obucheniya/#comment-157959

# HSE
http://wiki.cs.hse.ru/%D0%97%D0%B0%D0%B3%D0%BB%D0%B0%D0%B2%D0%BD%D0%B0%D1%8F_%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B8%D1%86%D0%B0

# Cheat Sheets
https://assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf
https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf

# machine-learning

## Courses
### Machine Learning Certification by Stanford University (Coursera)
https://www.coursera.org/learn/machine-learning?ranMID=40328&ranEAID=vedj0cWlu2Y&ranSiteID=vedj0cWlu2Y-7qVTEhC_VzBMKbCqkh1wKg&siteID=vedj0cWlu2Y-7qVTEhC_VzBMKbCqkh1wKg&utm_content=10&utm_medium=partners&utm_source=linkshare&utm_campaign=vedj0cWlu2Y

### Machine Learning Data Science Course from Harvard University (edX)
https://www.edx.org/professional-certificate/harvardx-data-science?source=aw&awc=6798_1558507227_53f78d4489236bf0ab037033bbd29b18&utm_source=aw&utm_medium=affiliate_partner&utm_content=text-link&utm_term=427859_Digital+Defynd

### Machine Learning A-Z™: Hands-On Python & R In Data Science (Udemy)
https://www.udemy.com/machinelearning/?ranMID=39197&ranEAID=vedj0cWlu2Y&ranSiteID=vedj0cWlu2Y-lDG4kOFfEfvaPGh0v0B4fg&LSNPUBID=vedj0cWlu2Y

### Mathematics for Machine Learning by Imperial College London (Coursera)
https://www.coursera.org/specializations/mathematics-machine-learning?ranMID=40328&ranEAID=vedj0cWlu2Y&ranSiteID=vedj0cWlu2Y-KCHg78lYrmaVFgGZHEyT1Q&siteID=vedj0cWlu2Y-KCHg78lYrmaVFgGZHEyT1Q&utm_content=10&utm_medium=partners&utm_source=linkshare&utm_campaign=vedj0cWlu2Y

### Advanced Machine Learning Course by HSE (Coursera)
https://www.coursera.org/specializations/aml?ranMID=40328&ranEAID=vedj0cWlu2Y&ranSiteID=vedj0cWlu2Y-WuX_GsDhGclKOiEvgv4cOQ&siteID=vedj0cWlu2Y-WuX_GsDhGclKOiEvgv4cOQ&utm_content=10&utm_medium=partners&utm_source=linkshare&utm_campaign=vedj0cWlu2Y

### Python for Data Science and Machine Learning Bootcamp (Udemy)
https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/?ranMID=39197&ranEAID=vedj0cWlu2Y&ranSiteID=vedj0cWlu2Y-If8FN0EIAIzcASogP0xxKg&LSNPUBID=vedj0cWlu2Y

### Machine Learning – Artificial Intelligence by Columbia University (edX)
https://www.edx.org/micromasters/columbiax-artificial-intelligence?source=aw&awc=6798_1558507517_b38fca3a4533ac00094824d55f62e9d3&utm_source=aw&utm_medium=affiliate_partner&utm_content=text-link&utm_term=427859_Digital+Defynd

### Machine Learning Courses for Beginners (LinkedIn Learning – Lynda)
https://linkedin-learning.pxf.io/c/1238999/449670/8005?subId1=ddmachinelearning1&u=https%3A%2F%2Fwww.linkedin.com%2Flearning%2Ftopics%2Fmachine-learning%3FentityType%3DCOURSE

### Machine Learning Certification by University of Washington (Coursera)
https://www.coursera.org/specializations/machine-learning?ranMID=40328&ranEAID=vedj0cWlu2Y&ranSiteID=vedj0cWlu2Y-qJFgc9g.r_2biUIa8roXbw&siteID=vedj0cWlu2Y-qJFgc9g.r_2biUIa8roXbw&utm_content=10&utm_medium=partners&utm_source=linkshare&utm_campaign=vedj0cWlu2Y


## Pandas Profiling
https://github.com/pandas-profiling/pandas-profiling

pd.set_option('display.max_colwidth',1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


data = pd.read_csv('../../datasets/titanic/train.csv')
data['Name'] = data['Name'].str.replace('\,|\.|\(|\)|Mr|Jr|Dr|Rev|\"|Master', '')
pd.value_counts(data[(data['Sex'] == 'male')]['Name'].apply(lambda x: x.split(' ')).sum(axis=0), sort=True).head(10)
