# create doc
echo 'create documentation'
pdoc --html -o . --template-dir ./config --force ../msmhelper

mv msmhelper/* .
rmdir msmhelper

# replace NP_DOC with link
sed -i -e 's/NP_DOC/https:\/\/docs.scipy.org\/doc\/numpy\/reference\/generated/g' *.html
echo ''
