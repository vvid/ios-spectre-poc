# ios-spectre-poc

## PURPOSE
  This is a proof of concept of Spectre v1 attack  (http://spectreattack.com).
It can only read data from the same process.
There are some code for Meltdown (http://meltdownattack.com) but it doesn't work, 
or I didn't tried hard enough.
The value of privileged register MIDR_EL1 (shown during startup) is a noise, at least in my test environment.

Tested on Apple A7, A9, A11 processors.

## USAGE
  Press Start and wait for log.
If it succeeded, there should be "Correct Horse Battery Staple?" text and little code dump.
