from setuptools import setup
setup(
    name='randomforestimputer',
    license='MIT License',
    packages=['randomforestimputer'],
    install_requires=['numpy','pandas','imblearn','sklearn','warnings'],
    version='0.0.1', 
    description='Building a imputer algorithm based on clustering property of random forest',
    author='Ngoc Nguyen',
    url='https://github.com/nbn14',
    download_url='https://github.com/nbn14/randomforestimputer',
    keywords=['data', 'statistical test', 'data analysis', 'random forest', 'missing data', 'imputation','data science', 'pandas', 'python'],
    classifiers=[]
)
