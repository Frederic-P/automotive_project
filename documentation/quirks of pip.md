- Some modules in pip have dependencies, these dependencies get auto-isntalled and seem to break the tensorflow-rocm module.
- Using the frozen requirements.txt file does not guarantee the correct environment will be set up; you need to specifically use the wheel package from AMD.com
- In case pip ruins the rocm-compatible version of tf, you can revert that with ` pip install tensorflow-rocm==2.17 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3 --upgrade  --force-reinstall `

