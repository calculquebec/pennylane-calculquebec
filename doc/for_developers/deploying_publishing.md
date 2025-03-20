# The deployment and publishing process

The aim of this file is to give a checklist to deploying and publishing a version of Pennylane-Snowflurry on Pypi, starting from the ```dev``` branch.

1. when you are ready to deploy and release, bump the version in the ```_version.py``` file, in the ```pennylane_calculquebec``` folder. The version number is composed of three distinct number. The logic goes as follows : 

    - if the new version is a patch or a minor change, bump the rightmost number.
    - if the new version is a big addition, bump the middle number, and set the rightmost number to 0.
    - if the new version makes the plugin go to a stable revision that is unlikely to change afterwards, bump the leftmost number and set the others to 0.

2. When the bumped version is in dev, create a pull request from dev to main, containing a list of the changes you made, accompanied with the version in the title. once the PR has been approved, merge it to main. 

3. add a tag to the ```main``` branch which matches the version you put in the ```_version.py``` file. You can create a new tag in most of the git graphical applications. If you're using a terminal, the commands are : 

```
git tag v0.0.0 (change the numbers for your version)
git push --tags
```

4. Create a new release ([doc](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)) and publish it. This will start a workflow which is responsible to build the python package and upload it to pypi with the documentation contained in the readme.md. Once the workflow is green, you're good to go!

