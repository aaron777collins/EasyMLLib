@echo off
del /s /f /q build\*.*
for /f %%f in ('dir /ad /b build\') do rd /s /q build\%%f
del /s /f /q dist\*.*
for /f %%f in ('dir /ad /b dist\') do rd /s /q dist\%%f
del /s /f /q EasyMLLIB.egg-info\*.*
for /f %%f in ('dir /ad /b EasyMLLIB.egg-info\') do rd /s /q EasyMLLIB.egg-info\%%f
python setup.py sdist bdist_wheel
twine upload dist/*