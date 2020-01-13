# Docker:

```
# Build the docker image.
docker build -t local:kanshi .

# Start the container  which
docker run -d --restart always -p 23334:8002 local:kanshi     # for production

```

```
# Below commands for other usage

# docker update --restart=no local:kanshi  # disable always restart

# docker run --rm -p 23334:8002 local:kanshi   # start for debugging
```


# Some sources:

- http://cls.hs.yzu.edu.tw/300/ALL/primary1/pr1.htm
- https://www.zhihu.com/question/22048471
- https://hslin.nidbox.com/diary/read/7032260
- https://zh.wikipedia.org/wiki/%E8%BF%91%E4%BD%93%E8%AF%97
- http://www.tufs.ac.jp/ts/personal/choes/etc/kansi/index.html



