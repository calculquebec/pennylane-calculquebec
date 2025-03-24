# The git flow

There are 2 important branches : 
-  **main** : it contains the version that's available to the public and should never be updated directly
-  **dev** : it contains the most up to date version, and is updated as often as developers add changes.

Those two branches are never pushed to directly. Developers work on **feature branches** which are created for specific tasks, and get deleted once the task is over and the feature branch is merged into the dev branch. 

In order to contribute to the dev branch, a developer should create a new feature branch, commit their work into it, and push it to github. Once this is done, said developer can create a new **pull request** from their branch to the dev branch. 

# The deployment and publishing process

The aim of this file is to give a checklist to deploying and publishing a version of Pennylane-CalculQuebec on Pypi, starting from the ```dev``` branch.

1. when you are ready to deploy and release, **bump the version** in the ```_version.py``` file, in the ```pennylane_calculquebec``` folder. The version number is composed of three distinct numbers. The logic goes as follows : 

    - if the new version is a patch or a minor change, bump the rightmost number.
    - if the new version is a big addition, bump the middle number, and set the rightmost number to 0.
    - if the new version makes the plugin go to a stable revision that is unlikely to change afterwards, bump the leftmost number and set the others to 0.

2. When the bumped version is in dev, **create a pull request** from dev to main. Put a list of the changes you made in the description, accompanied with the version in the title. once the PR has been approved, merge it to main. If it's a big change, you might need to get multiple reviews.

3. When dev is merged in main, **add a tag** to the ```main``` branch which matches the version you put in the ```_version.py``` file. You can create a new tag using most of existing git graphical applications. If you're using a terminal, the commands are : 

```
git tag v0.0.0 (change the numbers for your version)
git push --tags
```

4. Create a new release (by following the doc [here](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)), compare changes from the tag you created in last step to the tag of the previous release. Once you have done that, you can add any important information specific to the release and publish it. Doing so will start execution of a workflow which will automatically build the python package and upload it to pypi with the documentation contained in the readme.md. If the workflow turns green, it means it succeeded and you're good to go. If it turns red, it means something did not go as expected, and needs fixing. When have fixed the problem you'll have to create a new pull request from your patch branch to the dev branch, and start again from step 1. 
