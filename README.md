# MMPose教程

OpenMMLab 主页：https://openmmlab.com

MMPose主页：https://github.com/open-mmlab/mmpose

视频讲解：同济子豪兄 https://space.bilibili.com/1900783

环境：GPU RTX 3060、CUDA v11.8



1. 创建本地仓库

```
git init 
```

2. 将代码提交至暂存区(**git add .** 默认将所有文件提交至本地仓库)

   **注意：不要遗忘点和点前面的空格**

   ```
   git add .
   ```

   4. 输入指令 **`git commit -m’提交说明备注’`** 将暂存区代码提交至本地仓库
   5. 输入指令 **`git remote add origin + 远程仓库地址`** 绑定远程仓库

```
git remote add origin https://github.com/storm-ice/MMPose_Tutorials
```

```
git remote set-url origin https://github.com/storm-ice/MMPose_Tutorials
```

```
git remote -v
```


输入指令 git push -u origin master 将本地仓库推送至远程仓库
```
git push -u origin main
```



```bash
https://github.com/storm-ice/MMPose_DMS.git
```



```bash
echo "# MMPose_DMS" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/storm-ice/MMPose_DMS.git
git push -u origin main
```





```bash
git remote add origin https://github.com/storm-ice/MMPose_DMS.git
git branch -M main
git push -u origin main
```

