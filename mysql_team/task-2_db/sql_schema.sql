CREATE DATABASE IF NOT EXISTS `power_consumption` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE `power_consumption`;

USE `power_consumption`;

CREATE TABLE IF NOT EXISTS `power_consumption`.`time_dimension` (
    `time_id` INT NOT NULL AUTO_INCREMENT,
    `recorded_at` DATETIME NULL,
    `hour` TINYINT NULL,
    `day_of_week` TINYINT NULL,
    `month` TINYINT NULL,
    `year` SMALLINT NULL,
    `is_weekend` TINYINT NULL,
    PRIMARY KEY (`time_id`)
) ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `power_consumption`.`power_reading`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `power_consumption`.`power_reading` (
    `reading_id` BIGINT NOT NULL AUTO_INCREMENT,
    `global_active_power` DECIMAL(8, 3) NULL,
    `global_reactive_power` DECIMAL(8, 3) NULL,
    `voltage` DECIMAL(8, 3) NULL,
    `global_intensity` DECIMAL(8, 3) NULL,
    `time_id` INT NULL,
    PRIMARY KEY (`reading_id`),
    INDEX `time_id_idx` (`time_id` ASC) VISIBLE,
    CONSTRAINT `time_id` FOREIGN KEY (`time_id`) REFERENCES `power_consumption`.`time_dimension` (`time_id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `power_consumption`.`Sub_metering`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `power_consumption`.`Sub_metering` (
    `metering_id` INT NOT NULL AUTO_INCREMENT,
    `kitchen` DECIMAL(8, 3) NULL,
    `Laundry` DECIMAL(8, 3) NULL,
    `water_heater_ac` DECIMAL(8, 3) NULL,
    `reading_id` BIGINT NULL,
    PRIMARY KEY (`metering_id`),
    INDEX `reading_id_idx` (`reading_id` ASC) VISIBLE,
    CONSTRAINT `reading_id` FOREIGN KEY (`reading_id`) REFERENCES `power_consumption`.`power_reading` (`reading_id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `power_consumption`.`staging_raw` (
    `raw_date` VARCHAR(10),
    `raw_time` VARCHAR(8),
    `global_active_power` VARCHAR(10),
    `global_reactive_power` VARCHAR(10),
    `voltage` VARCHAR(10),
    `global_intensity` VARCHAR(10),
    `sub_metering_1` VARCHAR(10),
    `sub_metering_2` VARCHAR(10),
    `sub_metering_3` VARCHAR(10)
) ENGINE = InnoDB;