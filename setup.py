import setuptools

with open('requirements.txt') as open_file:
    install_requires = open_file.read()

setuptools.setup(
    name='marginap',
    version='',
    packages=[''],
    url='https://github.com/elias-ramzi/margin_ap',
    license='',
    author='Elias Ramzi',
    author_email='elias.ramzi@lecnam.net',
    description='',
    python_requires='>=3.6',
    install_requires=install_requires
)
