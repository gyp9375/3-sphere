@echo off
set VCPKG=C:\vcpkg
g++ -O2 -std=c++17 main.cpp -o s3_viewer.exe ^
  -I %VCPKG%\installed\x64-mingw-static\include ^
  -L %VCPKG%\installed\x64-mingw-static\lib ^
  -lglfw3 -lglad -lopengl32 -lgdi32 -luser32 -lshell32 ^
  -static-libgcc -static-libstdc++
if errorlevel 1 exit /b 1
echo Build OK
