from bs4 import BeautifulSoup
import requests
import os
import argparse

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import logging
from time import sleep
from urllib.error import URLError
from selenium.common.exceptions import WebDriverException, TimeoutException, NoSuchElementException
import shutil
import hashlib
import urllib.parse
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm


#Butterfly credentials (overridden by CLI args in main)
username = ''
password = ''
archivename = ''

#Download directory (overridden by CLI args in main)
download_dir = ''
videofolder = 'Uncleaned'


def extract_info_from_page():
    sleep(10)
    rows = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//tr[@data-bni-id='DataGridTableRow']")))
    for row in rows:
        try:
            cell = row.find_element(By.CLASS_NAME, 'DataGridTable-module--last-frozen--5Qjle')
            
            a_tag = cell.find_element(By.TAG_NAME, 'a')
            span_tag = cell.find_element(By.CLASS_NAME, "flex-grow.font-bold.truncate")
            
            href = a_tag.get_attribute('href')
            
            title = span_tag.text
            
            file_info.add((href, title))
        except Exception as e:
            print("An error occurred while extracting information from a row:", e)

def go_to_next_page():
    next_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='next-btn']")))
    next_button.click()

def extract_video_urls_from_page():
    groups = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//div[contains(@class, 'relative rounded group w-[100px] h-[100px] AspectRatioBox-CssProp1_Component-module--cls2--5tOys AspectRatioBox-CssProp1_Component-module--cls1--6B+CF ')]")))
    videos = []
    for group in groups:
        inset_div = group.find_element(By.XPATH, ".//div[contains(@class, 'inset-0 absolute')]")
        a_tag = inset_div.find_element(By.TAG_NAME, 'a')
        href = a_tag.get_attribute('href')
        videos.append(href)
    return videos

def check_if_draft():
    """Check if the current exam is in draft state"""
    try:
        draft_element = driver.find_element(By.XPATH, "//div[contains(text(), 'Draft')]")
        return True
    except NoSuchElementException:
        return False

def download_file_directly(url, filename):
    """Download file directly using requests with session cookies from selenium"""
    try:
        # Get cookies from selenium session
        selenium_cookies = driver.get_cookies()
        session = requests.Session()
        
        # Add selenium cookies to requests session
        for cookie in selenium_cookies:
            session.cookies.set(cookie['name'], cookie['value'])
        
        # Download the file
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        filepath = os.path.join(current_downloadspath, filename)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logging.info(f"Successfully downloaded file directly: {filename}")
        return True
        
    except Exception as e:
        logging.error(f"Error downloading file directly: {e}")
        return False

def extract_filename_from_url(url):
    """Extract filename from the download URL"""
    try:
        # Parse the URL to get query parameters
        parsed_url = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        # Look for filename in response-content-disposition
        if 'response-content-disposition' in query_params:
            content_disposition = query_params['response-content-disposition'][0]
            # Extract filename from content-disposition
            if 'filename=' in content_disposition:
                filename_part = content_disposition.split('filename=')[1].split(';')[0]
                # Remove quotes and decode
                filename = filename_part.strip('"').split('/')[-1]
                return filename
        
        # Fallback: extract from URL path
        path = parsed_url.path
        filename = path.split('/')[-1]
        if filename:
            return filename
        
        # Final fallback: generate a filename
        return f"video_{hash(url) % 100000}.mp4"
        
    except Exception as e:
        logging.error(f"Error extracting filename from URL: {e}")
        return f"video_{hash(url) % 100000}.mp4"

def download_video_draft(video_url):
    """Download video that is in draft state"""
    try:
        # Navigate to the video page
        driver.get(video_url)
        logging.info(f"Navigating to draft video page: {video_url}")
        
        # Wait for page to load
        sleep(3)
        
        # Find the download button and get its href
        download_button = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//a[@data-bni-id='DownloadButton']"))
        )
        
        download_url = download_button.get_attribute('href')
        if not download_url:
            logging.error("No download URL found in download button")
            return False
        
        logging.info(f"Found download URL: {download_url}")
        
        # Extract filename from URL
        filename = extract_filename_from_url(download_url)
        logging.info(f"Extracted filename: {filename}")
        
        # Download the file directly (synchronous - no need to wait for files to appear)
        success = download_file_directly(download_url, filename)
        
        if success:
            # Optionally verify the file exists (quick check)
            filepath = os.path.join(current_downloadspath, filename)
            if os.path.exists(filepath):
                logging.info(f"Successfully downloaded draft video: {filename}")
                return True
            else:
                logging.error(f"File not found after download: {filename}")
                return False
        else:
            logging.error(f"Failed to download draft video from {video_url}")
            return False
            
    except Exception as e:
        logging.error(f"Error downloading draft video {video_url}: {e}")
        return False

def download_video_signed(video_url):
    """Download video that is signed (original method)"""
    try:
        # Navigate to the video page
        driver.get(video_url)
        logging.info(f"Navigating to signed video page: {video_url}")
        
        # Wait for the page to fully load
        sleep(3)
        
        # First, look for the download button
        download_button = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@data-bni-id='DownloadButton']"))
        )
        logging.info("Found download button")
        download_button.click()
        logging.info("Clicked download button")
        
        # Then look for the submit/confirm button and click it
        submit_button = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
        )
        logging.info("Found submit button")
        submit_button.click()
        logging.info("Clicked submit button")
        
        # Wait for download to start and complete
        initial_files = set(os.listdir(current_downloadspath))
        
        # Wait for new files to appear (up to 30 seconds)
        max_wait = 30
        for i in range(max_wait):
            sleep(1)
            current_files = set(os.listdir(current_downloadspath))
            new_files = current_files - initial_files
            if new_files:
                logging.info(f"Download completed after {i+1} seconds")
                break
            if i == max_wait - 1:
                logging.warning("No new files detected after maximum wait time")
        
        # Check if download was successful
        if len(os.listdir(current_downloadspath)) > len(initial_files):
            logging.info(f"Successfully downloaded signed video from {video_url}")
            return True
        else:
            logging.warning(f"No new files found in download directory for {video_url}")
            return False
            
    except Exception as e:
        logging.error(f"Error during signed video download process: {e}")
        return False

def download_video(videos):
    """Enhanced download function that handles both draft and signed videos with fallback"""
    for video_url in videos:
        try:
            # Navigate to the video page first to check status
            driver.get(video_url)
            logging.info(f"Navigating to video page: {video_url}")
            sleep(3)
            
            # Check if video is in draft state
            is_draft = check_if_draft()
            
            success = False
            
            if is_draft:
                logging.info("Video is in draft state, using direct download method")
                success = download_video_draft(video_url)
            else:
                logging.info("Video is signed, using original download method")
                success = download_video_signed(video_url)
                
                # If signed method failed, try draft method as fallback
                if not success:
                    logging.warning("Signed download failed, attempting fallback to direct download method")
                    try:
                        success = download_video_draft(video_url)
                        if success:
                            logging.info("Fallback to direct download succeeded!")
                        else:
                            logging.error("Both signed and direct download methods failed")
                    except Exception as fallback_error:
                        logging.error(f"Fallback download method also failed: {fallback_error}")
            
            if not success:
                logging.error(f"Failed to download video from {video_url}")
                
        except Exception as e:
            logging.error(f"Error downloading {video_url}: {e}")
            # Try direct download as last resort
            logging.warning("Attempting direct download as last resort...")
            try:
                success = download_video_draft(video_url)
                if success:
                    logging.info("Last resort direct download succeeded!")
            except Exception as last_resort_error:
                logging.error(f"Last resort download failed: {last_resort_error}")

def download_and_rename_files(file_info, videofolderpath, current_downloadspath, max_retries=1):
    failed_downloads = []
    completed_downloads = []
    
    for href, title in tqdm(file_info, desc="Processing Files", unit="file", ncols=100):
        retries = 0
        if title:
            safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c in ' -_']).rstrip()
            
            # Check if this patient/exam already exists
            patient_folder = os.path.join(videofolderpath, safe_title)
            if os.path.exists(patient_folder):
                logging.info(f"Skipping {safe_title} - already exists")
                continue
                
        while retries < max_retries:
            try:
                driver.get(href)
                vids = extract_video_urls_from_page()
                download_video(vids)  # Now handles both draft and signed videos
                break  
            except (URLError, WebDriverException) as e:
                logging.error(f"Error downloading {title} from {href}: {e}")
                retries += 1
                sleep(15)  
                continue
                
        # Check if any files were actually downloaded
        downloaded_files = os.listdir(current_downloadspath)
        if not downloaded_files:
            logging.warning(f"No files downloaded for {title}")
            failed_downloads.append((href, title))
            continue
            
        # Rename and move files
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c in ' -_']).rstrip()
        new_path = os.path.join(videofolderpath, safe_title)
        
        try:
            if not os.path.exists(new_path):
                os.rename(current_downloadspath, new_path)
                logging.info(f"Successfully moved files to: {new_path}")
            else:
                logging.warning(f"Directory {new_path} already exists, merging files")
                # Move files individually if directory exists
                for file in downloaded_files:
                    src = os.path.join(current_downloadspath, file)
                    dst = os.path.join(new_path, file)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                shutil.rmtree(current_downloadspath)
                
        except Exception as move_error:
            logging.error(f"Error moving files for {title}: {move_error}")
            failed_downloads.append((href, title))
            
        # Recreate current_downloads directory for next iteration
        os.makedirs(current_downloadspath, exist_ok=True)
        completed_downloads.append((href, title))
        logging.info(f"Prepared {current_downloadspath} for the next download.")
    
    logging.info("Finished download process")
    print(f"Failed downloads: {len(failed_downloads)}")
    print(f"Completed downloads: {len(completed_downloads)}")
    return failed_downloads, completed_downloads

def init_session(download_directory, video_folder, headless=False):
    """Initialize download paths, Chrome driver, and wait objects."""
    # Create all directories if they don't exist (including parent directories)
    videofolderpath_local = os.path.join(download_directory, video_folder)
    os.makedirs(videofolderpath_local, exist_ok=True)

    current_downloadspath_local = os.path.join(videofolderpath_local, 'current_downloads')
    os.makedirs(current_downloadspath_local, exist_ok=True)

    options = Options()
    if headless:
        # Use new headless mode if available
        options.add_argument("--headless=new")
    prefs = {"download.default_directory": current_downloadspath_local, "download.prompt_for_download": False}
    options.add_experimental_option('prefs', prefs)

    driver_local = webdriver.Chrome(options=options)
    wait_local = WebDriverWait(driver_local, 20)

    return driver_local, wait_local, videofolderpath_local, current_downloadspath_local

def login_and_open_archive(driver_local, wait_local, user, pwd, archive_name):
    """Log into Butterfly Cloud and open the specified archive."""
    driver_local.get("https://cloud.butterflynetwork.com/")

    email_field = wait_local.until(EC.presence_of_element_located((By.XPATH, "//input[@data-bni-id='emailField']")))
    email_field.clear()
    email_field.send_keys(user)

    password_field = driver_local.find_element(By.XPATH, "//input[@data-bni-id='passwordField']")
    password_field.clear()
    password_field.send_keys(pwd)

    login_button = driver_local.find_element(By.XPATH, "//button[@data-bni-id='loginButton']")
    login_button.click()

    link = WebDriverWait(driver_local, 20).until(
        EC.presence_of_element_located(
            (By.XPATH, "//span[contains(text(), '{}')]/ancestor::a".format(archive_name))
        )
    )
    href = link.get_attribute('href')
    link.click()
    return href

def run(user, pwd, archive_name, download_directory, video_folder, headless=False, ids_to_gather=None):
    """Execute the full scraping and download workflow."""
    # Make globals available to helper functions defined above
    global driver, wait, videofolderpath, current_downloadspath, file_info

    driver, wait, videofolderpath, current_downloadspath = init_session(download_directory, video_folder, headless=headless)
    try:
        login_and_open_archive(driver, wait, user, pwd, archive_name)

        # Collecting info from all pages (shared with extract_info_from_page)
        file_info = set()
        while True:
            extract_info_from_page()
            try:
                go_to_next_page()
            except Exception as e:
                print("No more pages or an error occurred:", e)
                break

        # Determine which files to download
        new_file_info = set()
        if not ids_to_gather:
            new_file_info = file_info
            logging.info(f"No specific IDs provided, downloading all {len(file_info)} files")
        else:
            for f in file_info:
                if f[1].split(',')[0] in ids_to_gather:
                    new_file_info.add(f)
            logging.info(f"Filtered to {len(new_file_info)} files based on ID list")

        print(f"Files to download: {len(new_file_info)}")
        print(f"Total files found: {len(file_info)}")
        failed, completed = download_and_rename_files(new_file_info, videofolderpath, current_downloadspath)
        return failed, completed
    finally:
        try:
            driver.quit()
        except Exception:
            pass

def parse_args():
    parser = argparse.ArgumentParser(description="Scrape and download videos from Butterfly Cloud archive")
    parser.add_argument("--username", required=True, help="Butterfly Cloud username (email)")
    parser.add_argument("--password", required=True, help="Butterfly Cloud password")
    parser.add_argument("--archive", required=True, help="Archive name to open")
    parser.add_argument("--download-dir", required=True, help="Base directory to store downloads")
    parser.add_argument("--videofolder", default="Uncleaned", help="Subfolder name under download-dir")
    parser.add_argument("--ids", default="", help="Comma-separated list of IDs to download (optional)")
    parser.add_argument("--headless", action="store_true", help="Run Chrome in headless mode")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ids_list = [i.strip() for i in args.ids.split(',') if i.strip()] if args.ids else None
    run(
        user=args.username,
        pwd=args.password,
        archive_name=args.archive,
        download_directory=args.__dict__["download_dir"],
        video_folder=args.videofolder,
        headless=args.headless,
        ids_to_gather=ids_list,
    )
    
#     python UltrAI/data/butterflycloudscraper.py \
#   --username "<email>" \
#   --password "<password>" \
#   --archive "<archive name as shown in UI>" \
#   --download-dir "/absolute/path/where/to/save" \
#   --videofolder "Uncleaned" \
#   --headless