# create doc
echo 'create documentation'
pdoc3 --html -o . --template-dir . --force ../

mv msmhelper/* .
rmdir msmhelper

# replace NP_DOC with link
sed -i -e 's/NP_DOC/https:\/\/docs.scipy.org\/doc\/numpy\/reference\/generated/g' *.html
echo ''
