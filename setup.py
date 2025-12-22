from setuptools import setup, find_packages

setup(name='lipnet',
    version='0.1.6',
    description='End-to-end sentence-level lipreading',
    url='http://github.com/rizkiarm/LipNet',
    author='Muhammad Rizki A.R.M',
    author_email='rizki@rizkiarm.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
	install_requires=[
        'tensorflow>=2.10.0',
        'keras>=2.10.0',
        'editdistance>=0.6.0',
        'h5py>=3.7.0',
        'matplotlib>=3.5.0',
        'numpy>=1.23.0',
        'python-dateutil>=2.8.0',
        'scipy>=1.9.0',
        'Pillow>=9.0.0',
        'nltk>=3.7',
        'sk-video>=1.1.10',
        'opencv-python>=4.6.0'
    ])
