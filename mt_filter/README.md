# MT_filter
NLP sensitive/bad/profane words detect.



使用示例： 



```python
from sensewords import ChineseFilter, EnglishFilter

ChineseFilter.filter("真气人，他妈的")
# Out[3]: {'1': '他妈的'} # '1'代表有敏感字，'0'代表无
ChineseFilter.filter("你好啊")
# Out[4]: {'0': ''}


EnglishFilter.filter('fuck you baby')
# out[6]: {'1': 'fuck'}
EnglishFilter.filter('hello baby')
# Out[7]: {'0': ''} 

ChineseFilter.filter("中国共产党二十大胜利召开")
# Out[3]: {'1': '二十大'}

```

'''


