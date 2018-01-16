---
layout: post
title: The Not-So-Slow dance of Python and SQLServer
img: cntk.jpg
date: 2017-12-12 12:55:00 +0000
description: They're in love
tag: [sql, sqlserver, python]
comments: true
---

As I follow along [this]() getting-started tutorial, I do begin to get excited as I set up my Linux (Ubuntu) Docker container with SQLServer image living inside.  I'm also itching to get started creating some connections with `pyodbc`.  Why is this so feverishly exciting?  Because of `dask` out-of-core data frames and `pyarrow` Apache Arrow columnar/in-memory efficient tables of course.

Run the handy SQL command tool: 

`sqlcmd -S 127.0.0.1 -U sa -P $SQL_PASS`

where `$SQL_PASS` is the SQL Server password as an environment variable.

Then, one line at a time, enter:

```sql
USE SampleDB;
INSERT INTO Employees (Name, Location) VALUES (N'Alice', N'Australia'), (N'Mad Hatter', N'England'), (N'White Rabbit', N'India');
GO;
```

Few snags:


2.  ODBC connections are looking for the ODBC driver for SQL Server in the wrong place, so used `odbcinst -j` to find out where it was looking then symlink'd to that location as in [this SO](https://stackoverflow.com/questions/44527452/cant-open-lib-odbc-driver-13-for-sql-server-sym-linking-issue).

Problems:
- No extra word "DATABASE" needed when using "USE"
- mapping to "localhost" isn't working, only 127.0.0.1
- Spelling error:  Successfuly
