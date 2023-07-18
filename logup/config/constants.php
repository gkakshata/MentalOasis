<?php

//Start Session

session_start();
$errors = array();

//Create Constants to store Non Repeating Values
define("SITEURL",'http://localhost/dbb/');

define('LOCALHOST', 'localhost');

define('DB_USERNAME', 'root');

define('DB_PASSWORD', '');

define('DB_NAME', 'user');

$conn = mysqli_connect(LOCALHOST, DB_USERNAME, DB_PASSWORD) or die(mysqli_error()); //Database Connection
$db_select = mysqli_select_db($conn, DB_NAME) or die(mysqli_error()); //SElecting Database
?>