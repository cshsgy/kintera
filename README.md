# KINTERA

### Automatic fetched dependencies
During the build process, the following dependencies will be fetched and cached automatically:
| Package Name | Repository URL | Version |
|-------------|---------------|---------|
| fmt | [https://github.com/fmtlib/fmt](https://github.com/fmtlib/fmt) | 11.1.2 |
| yaml-cpp | [https://github.com/jbeder/yaml-cpp](https://github.com/jbeder/yaml-cpp) | 0.8.0 |
| elements | [https://github.com/chengcli/elements](https://github.com/chengcli/elements) | v1.1 |
| gtest | [https://github.com/google/googletest](https://github.com/google/googletest) | v1.13.0 |
| pyharp | [https://github.com/chengcli/pyharp](https://github.com/chengcli/pyharp) | v1.1.1 |
| disort | [https://${ACCOUNT}:${TOKEN}@github.com/zoeyzyhu/pydisort](https://${ACCOUNT}:${TOKEN}@github.com/zoeyzyhu/pydisort) | 24569ab591dc |


The cache is stored in the `.cache` directory and can be safely deleted at any time.
Internet connection is required for the first build and whenever after the cache is deleted.
Once the cache is populated, the build process can be repeated offline.

### Staying Updated
If you have forked this repository, please enable notifications or watch for updates!
