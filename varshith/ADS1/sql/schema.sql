-- =========================================
-- EMPLOYEE MANAGEMENT DATABASE
-- =========================================

CREATE DATABASE IF NOT EXISTS employee_management_system;

USE employee_management_system;

-- =========================================
-- 1. PERSONAL DETAILS TABLE
-- =========================================

CREATE TABLE IF NOT EXISTS employee_personal_details (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100),
    gender ENUM('Male', 'Female', 'Other'),
    date_of_birth DATE,
    age INT,
    personal_email VARCHAR(150) UNIQUE,
    office_email VARCHAR(150) UNIQUE,
    personal_phone VARCHAR(20),
    emergency_contact VARCHAR(20),
    father_name VARCHAR(150),
    mother_name VARCHAR(150),
    blood_group VARCHAR(10),
    marital_status ENUM('Single', 'Married', 'Divorced', 'Widowed'),
    aadhaar_number VARCHAR(20) UNIQUE,
    pan_number VARCHAR(20) UNIQUE,
    nationality VARCHAR(100) DEFAULT 'Indian',
    current_address TEXT,
    permanent_address TEXT,
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100),
    pincode VARCHAR(20),
    qualification VARCHAR(200),
    skills TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =========================================
-- 2. OFFICE DETAILS TABLE
-- =========================================

CREATE TABLE IF NOT EXISTS employee_office_details (
    office_record_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT,
    employee_code VARCHAR(50) UNIQUE,
    department VARCHAR(100),
    designation VARCHAR(100),
    employment_type ENUM('Full-Time', 'Part-Time', 'Intern', 'Contract'),
    work_location VARCHAR(150),
    manager_name VARCHAR(150),
    joining_date DATE,
    probation_end_date DATE,
    total_years_experience DECIMAL(4,1),
    years_in_company DECIMAL(4,1),
    salary_per_month DECIMAL(12,2),
    bonus DECIMAL(12,2) DEFAULT 0,
    shift_type VARCHAR(50),
    work_mode ENUM('Remote', 'Hybrid', 'Office'),
    official_status ENUM('Active', 'On Leave', 'Resigned', 'Terminated') DEFAULT 'Active',
    last_promotion_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_office_employee
        FOREIGN KEY (employee_id)
        REFERENCES employee_personal_details(employee_id)
        ON DELETE CASCADE
);

-- =========================================
-- 3. SALARY & LEAVE TABLE
-- =========================================

CREATE TABLE IF NOT EXISTS employee_salary_leave_details (
    salary_record_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT,
    salary_month VARCHAR(20),
    total_working_days INT,
    present_days INT,
    leave_days INT DEFAULT 0,
    unpaid_leave_days INT DEFAULT 0,
    paid_leave_days INT DEFAULT 0,
    overtime_hours DECIMAL(5,2) DEFAULT 0,
    monthly_salary DECIMAL(12,2),
    leave_deduction DECIMAL(12,2) DEFAULT 0,
    tax_deduction DECIMAL(12,2) DEFAULT 0,
    pf_deduction DECIMAL(12,2) DEFAULT 0,
    bonus_amount DECIMAL(12,2) DEFAULT 0,
    final_salary DECIMAL(12,2),
    payment_status ENUM('Pending', 'Paid') DEFAULT 'Pending',
    payment_date DATE,
    remarks TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_salary_employee
        FOREIGN KEY (employee_id)
        REFERENCES employee_personal_details(employee_id)
        ON DELETE CASCADE
);

-- =========================================
-- 4. ATTENDANCE TABLE
-- =========================================

CREATE TABLE IF NOT EXISTS employee_attendance (
    attendance_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT,
    attendance_date DATE,
    check_in TIME,
    check_out TIME,
    total_hours DECIMAL(4,2),
    attendance_status ENUM('Present', 'Absent', 'Leave', 'Half-Day'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_attendance_employee
        FOREIGN KEY (employee_id)
        REFERENCES employee_personal_details(employee_id)
        ON DELETE CASCADE
);

-- =========================================
-- 5. DOCUMENTS TABLE
-- =========================================

CREATE TABLE IF NOT EXISTS employee_documents (
    document_id INT PRIMARY KEY AUTO_INCREMENT,
    employee_id INT,
    document_type VARCHAR(100),
    document_number VARCHAR(100),
    file_path TEXT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_document_employee
        FOREIGN KEY (employee_id)
        REFERENCES employee_personal_details(employee_id)
        ON DELETE CASCADE
);

-- =========================================
-- SAMPLE INSERT DATA
-- =========================================

INSERT INTO employee_personal_details (
    first_name, last_name, gender, date_of_birth, age, personal_email,
    personal_phone, emergency_contact, father_name, mother_name, blood_group,
    marital_status, current_address, city, state, country, qualification, skills
)
SELECT
    'Varshith', 'Thungapalli', 'Male', '2004-05-10', 22, 'varshith@gmail.com',
    '9876543210', '9876543211', 'Ramesh', 'Lakshmi', 'O+', 'Single',
    'Hyderabad', 'Hyderabad', 'Telangana', 'India', 'B.Tech AI & DS',
    'Python, AI, ML, LangChain, LangGraph'
WHERE NOT EXISTS (
    SELECT 1 FROM employee_personal_details WHERE personal_email = 'varshith@gmail.com'
);

INSERT INTO employee_office_details (
    employee_id, employee_code, department, designation, employment_type,
    work_location, manager_name, joining_date, total_years_experience,
    years_in_company, salary_per_month, bonus, shift_type, work_mode
)
SELECT
    employee_id, 'EMP001', 'AI Research', 'AI Engineer', 'Full-Time',
    'Hyderabad', 'Rahul Sharma', '2025-01-10', 2.5, 1.0,
    85000, 10000, 'Day Shift', 'Hybrid'
FROM employee_personal_details
WHERE personal_email = 'varshith@gmail.com'
AND NOT EXISTS (
    SELECT 1 FROM employee_office_details WHERE employee_code = 'EMP001'
);

INSERT INTO employee_salary_leave_details (
    employee_id, salary_month, total_working_days, present_days, leave_days,
    unpaid_leave_days, paid_leave_days, overtime_hours, monthly_salary,
    leave_deduction, tax_deduction, pf_deduction, bonus_amount, final_salary,
    payment_status, payment_date
)
SELECT
    employee_id, 'May-2026', 26, 24, 2, 1, 1, 10, 85000,
    3000, 5000, 2000, 10000, 85000, 'Paid', '2026-05-31'
FROM employee_personal_details
WHERE personal_email = 'varshith@gmail.com'
AND NOT EXISTS (
    SELECT 1 FROM employee_salary_leave_details
    WHERE employee_id = employee_personal_details.employee_id
    AND salary_month = 'May-2026'
);

-- =========================================
-- VIEW ALL EMPLOYEE DETAILS
-- =========================================

SELECT
    epd.employee_id,
    epd.first_name,
    epd.last_name,
    eod.department,
    eod.designation,
    eod.salary_per_month,
    esld.final_salary
FROM employee_personal_details epd
JOIN employee_office_details eod
    ON epd.employee_id = eod.employee_id
JOIN employee_salary_leave_details esld
    ON epd.employee_id = esld.employee_id;
