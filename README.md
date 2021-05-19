# mbed HW3
## NON mode
1. the mode will be initialized to NON mode at first
2. NON mode doesn't do anything.

## gesture UI mode
1. blink led 5 times -> gesture mode start
2. Use gesture to select angle.
    * screen will show ltr or rtl
    * ltr: left to right -> angle decrease
    * rtl: right to left -> angle increase
![](https://i.imgur.com/qMTFx6P.jpg)
![](https://i.imgur.com/kcxKdj9.jpg)

3. Use User button to return selected angle

## detect mode
1. blink led 3 times -> detect mode start
2. return angle every 100ms 
3. if reach target angle -> set to NON mode