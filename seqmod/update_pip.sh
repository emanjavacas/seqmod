
python setup.py sdist bdist_wheel
echo "twine register dist/[project-version].tar.gz"
