<h1 align="center"> IsCopy/你抄了吗 </h1>
<p align="center">
	<img src=".github/pic/logo.png" alt="logo"/><br/>
    <img src="https://img.shields.io/badge/python-3.6-blue.svg" alt="python"/>
    <img src="https://img.shields.io/github/last-commit/WhiteRobe/iscopy.svg" alt="last-commit"/>
    <a href="https://github.com/WhiteRobe/iscopy/blob/master/LICENSE">
    	<img src="https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000" alt="LICENSE"/>
    </a>
</p>

> 这是一个开发中的代码重复度检查框架。
>
> 目前版本仅实现了最基础的代码查重方式。

## 依赖 | Dependency

目前的依赖：
- pandas
- pygments

## 快速上手 | Quick Start

```shell
python anly.py --input ./demo --filename demo-data.py
```

## 架构 | Struture

- `analyser`：主要的分析器
- `extractor`： 特征提取器
- `purifier`：代码规格化