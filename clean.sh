kill -9 $(lsof -t -i tcp:4101) & # if running default for MacOS Monterey
kill -9 $(lsof -t -i tcp:4102) &
kill -9 $(lsof -t -i tcp:4105)