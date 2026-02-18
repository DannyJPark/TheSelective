__version__ = '1.0.0'
URL = 'https://github.com/papersubmissiononly-abc/TheSelective'

from setuptools import setup, find_packages

setup(name='TheSelective',
      version=__version__,
      description='TheSelective: Dual-Head Diffusion for Selective Molecule Generation',
      url=URL,
      license='MIT',
      download_url=f'{URL}/archive/{__version__}.tar.gz',
      keywords=['pytorch', 'drug design', 'diffusion model', 'selectivity', 'molecule generation'],
      install_requires=['biopython'],
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'ts_gen = scripts.sample_diffusion:main',
              'ts_gen4poc = scripts.sample_for_pocket:main',
              'ts_train = scripts.train_diffusion:main',
              'ts_eval = scripts.evaluate_diffusion:main',
          ],
      },
      zip_safe=False)
