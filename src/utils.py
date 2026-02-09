"""
General utility functions for everything and some global variables
"""

import torch
import numpy as np
import random
import os
import textwrap
import json


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                             General Global Variables           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region
GLOBAL_PROJECT_DIR = "/weka/home/mohata1/scratchcpriebe1/MO/network_manifolds/"



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                             Basic Utility Functions        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region

def set_seed(seed):
    """set the random seed everywhere relevant"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    return

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if \
                          torch.backends.mps.is_available() else "cpu")
    return device

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                    Logging
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region
def make_logfile(filepath):
    """opens a file (that will need to be closed)"""
    logfile = open(filepath, 'a')
    return logfile

def save_config(filepath, config:dict):
    """saves the config dictionary to a file"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    return

def print_and_write(message, *files):
    """prints message to console and writes to given files. 
    (files should be open)
    """
    print(message)
    for file in files:
        if file is not None:
            file.write(message+'\n')
    return

def close_files(*files):
    """closes files"""
    for file in files:
        if file is not None:
            file.close()
    return

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                              Value/Quantity Tracking
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region

def nice_interval(n: int):
    """Determines if an integer is nice. Nice integers are of the form
        {1, 2, 5} x 10^k
    """
    if n == 0:
        return True
    else:
        pwr = 10 ** int(np.log10(n))
        nice = True if (n == 0 or (n % pwr == 0 and n // pwr in (1, 2, 5))) \
            else False
        return nice


class AverageMeter(object):
    def __init__(self, name=None, format=':.2f'):
        self.reset()
        self.name = name
        self.format = format
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.format + '} ({avg' + self.format + '})'
        return fmtstr.format(**self.__dict__)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                            Errors and Validation        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region
def check_if_null(named_param_or_var, dne_alt, exists_alt=None):
    """ 
    checks if an object exists and returns the given alternate if not
    option to return a separate value if the object does exist also

    :param named_param_or_var: the object to check if None
    :param dne_alt: alternate value if it is None
    :param exists_alt: alternate value if it exists (default: None returns the 
                       original object)
    """
    exists_alt = exists_alt if exists_alt is not None else named_param_or_var
    return exists_alt if named_param_or_var is not None else dne_alt

def is_int_or_int_string(x):
    if isinstance(x, int):
        return True
    if isinstance(x, str) and x.isdigit():
        return True
    return False

def display_message(msg:str):
    return textwrap.fill(textwrap.dedent(msg))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                              (Basic) Mail Alerts          
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region
"""(main machine does not have mail set up for alerts as far as I can tell) so
these are useful if the same is the case for you. Configured only for gmail 
sending right now."""
import smtplib
from email.message import EmailMessage



def get_mail_login(infofile):
    """infofile should contain the username on the first line and the (app) 
    password on the second"""
    with open(infofile, "r") as file:
        lines = file.readlines()
    username, password = lines[0].strip(), lines[1].strip()
    return username, password


def gmail_login(username, password):
    """logs in to the gmail smtp server"""
    server = smtplib.SMTP_SSL(host="smtp.gmail.com", port=465)
    server.login(username, password)
    return server

# WISHLIST more login options


def compose_email(subject, content):
    """compose an email"""
    email = EmailMessage()
    email.set_content(content)
    email["Subject"] = subject
    return email

# --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -
# Email Config  
# --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -
EMAIL_FILE = "../mail.log"
EMAIL_USER, EMAIL_PASS = get_mail_login(EMAIL_FILE)

def DEFAULT_SERVER():
    return gmail_login(EMAIL_USER, EMAIL_PASS)

# --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -

def send_email(email:EmailMessage, send_to=EMAIL_USER, send_from=EMAIL_USER, 
               server=DEFAULT_SERVER(), close_connection=True):
    """send an email"""
    email["From"] = send_from
    email["To"] = send_to
    server.send_message(email)
    server.quit()
    print(f"Email {email['Subject']} sent to {send_to}")
    return


def compose_and_send_email(content, subject=None, send_to=EMAIL_USER, 
                           send_from=EMAIL_USER, server=DEFAULT_SERVER()):
    email = compose_email(subject, content)
    send_email(email, send_to=send_to, send_from=send_from, server=server)
    return
