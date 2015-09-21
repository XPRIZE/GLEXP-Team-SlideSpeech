from google.appengine.ext import ndb
from flask import Flask, redirect, jsonify
app = Flask(__name__)
app.config['DEBUG'] = True

# Note: We don't need to call run() since our application is embedded within
# the App Engine WSGI application server.

class Word(ndb.Model):
    """Sub model for representing a word."""
    word = ndb.StringProperty(indexed=False)
    phonemes = ndb.StringProperty(indexed=False)

@app.route('/lookup/<letters>')
def lookup(letters):
    """Return a list of words starting with given letters."""
    query = Word.query()
    words = []
    for word in query.fetch():
    	if word.word.startswith(letters):
	    	words.append(word.word)
    return jsonify({"words": words})

@app.route('/add')
def add():
    """Add some data"""
    data = """PUEBLO,C16/P C36/U V05/E C17/B C37/L V08/O 
PUEBLA,C16/P C36/U V06/E C17/B C37/L V04/A 
PLATTE,C16/P C37/L V01/A_E C18/TT 
PLATFORM,C16/P C37/L V01/A C18/T C24/F V11/O C38/R C33/M 
PLATITUDE,C16/P C37/L V01/A C18/T V02/I C18/T V09/U_E C19/D 
PLATYPUS,C16/P C37/L V01/A C18/T V04/Y C16/P V12/U C28/S 
PLATEN,C16/P C37/L V01/A C18/T V04/E C34/N 
"""
    for item in data.split("\n"):
    	items = item.split(',')
    	if len(items) == 2:
	    	new_word = Word(word=items[0], phonemes=items[1])
    		new_word.put()
    return "Data Added"

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    # return "Hello GDG Auckland"
    return redirect("/js/map_test.html", code=302)


@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return 'Sorry, nothing at this URL.', 404
