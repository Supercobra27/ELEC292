USE employees;

-- Question 1
SELECT * FROM employees;

-- Question 2
SELECT * FROM salaries WHERE salary * 1.7 > 100000;

-- Example 1
SELECT COUNT(DISTINCT last_name) FROM employees;

-- Question 3
SELECT AVG(salary) FROM salaries WHERE emp_no > 1510;

-- Example 2
SELECT last_name, COUNT(last_name) FROM employees GROUP BY last_name;

-- Question 4
SELECT emp_no, AVG(salary) FROM salaries GROUP BY emp_no;

-- Question 5
SELECT employees.first_name, employees.last_name, salaries.salary FROM employees INNER JOIN salaries ON employees.emp_no = salaries.emp_no; 

-- was getting errors
DROP PROCEDURE IF EXISTS emp_salary;

-- Example 3
DELIMITER $$
CREATE PROCEDURE emp_salary(IN p_emp_no INT)
BEGIN
SELECT
employees.emp_no, employees.first_name, employees.last_name, salaries.salary
FROM employees JOIN salaries ON employees.emp_no=salaries.emp_no
WHERE employees.emp_no = p_emp_no;
END $$
DELIMITER ;

call emp_salary(11300);

-- Question 6
DELIMITER $$
CREATE PROCEDURE emp_avg_salary(IN p_emp_no INT)
BEGIN
SELECT salaries.emp_no, AVG(salaries.salary) FROM salaries 
WHERE salaries.emp_no = p_emp_no
GROUP BY salaries.emp_no;
END $$
DELIMITER ;

call emp_avg_salary(11300);