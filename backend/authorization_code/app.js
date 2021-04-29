var express = require('express'); // Express web server framework
var request = require('request'); // "Request" library
var cors = require('cors');
var querystring = require('querystring');
var cookieParser = require('cookie-parser');
const { features } = require('process');

var client_id = 'd55ac39db811476c88d5e80051cca636'; // Your client id
var client_secret = '8aa2c23cfbc44ab8932c6a8b852564fa'; // Your secret
var redirect_uri = 'http://localhost:8888/callback'; // Your redirect uri
var access_token;
var refresh_token;
/**
 * Generates a random string containing numbers and letters
 * @param  {number} length The length of the string
 * @return {string} The generated string
 */
var generateRandomString = function (length) {
  var text = '';
  var possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

  for (var i = 0; i < length; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
};

var stateKey = 'spotify_auth_state';

var app = express();

app.use(express.static(__dirname + '/public'))
  .use(cors())
  .use(cookieParser());

app.get('/login', function (req, res) {

  var state = generateRandomString(16);
  res.cookie(stateKey, state);

  // your application requests authorization

  var scope = 'user-read-private user-read-email playlist-read-collaborative playlist-read-private user-library-read user-top-read user-follow-read user-library-modify user-library-read';
  res.redirect('https://accounts.spotify.com/authorize?' +
    querystring.stringify({
      response_type: 'code',
      client_id: client_id,
      scope: scope,
      redirect_uri: redirect_uri,
      state: state
    }));
});

app.get('/callback', function (req, res) {

  // your application requests refresh and access tokens
  // after checking the state parameter

  var code = req.query.code || null;
  var state = req.query.state || null;
  var storedState = req.cookies ? req.cookies[stateKey] : null;

  if (state === null || state !== storedState) {
    res.redirect('/#' +
      querystring.stringify({
        error: 'state_mismatch'
      }));
  } else {
    res.clearCookie(stateKey);
    var authOptions = {
      url: 'https://accounts.spotify.com/api/token',
      form: {
        code: code,
        redirect_uri: redirect_uri,
        grant_type: 'authorization_code'
      },
      headers: {
        'Authorization': 'Basic ' + (new Buffer.from(client_id + ':' + client_secret).toString('base64'))
      },
      json: true
    };

    request.post(authOptions, function (error, response, body) {
      if (!error && response.statusCode === 200) {
        access_token = body.access_token;
        refresh_token = body.refresh_token;

        var options = {
          url: 'https://api.spotify.com/v1/me',
          headers: { 'Authorization': 'Bearer ' + access_token },
          json: true
        };

        // use the access token to access the Spotify Web API
        request.get(options, function (error, response, body) {
          console.log(body);
        });

        // we can also pass the token to the browser to make requests from there
        res.redirect('/#' +
          querystring.stringify({
            access_token: access_token,
            refresh_token: refresh_token
          }));
      } else {
        res.redirect('/#' +
          querystring.stringify({
            error: 'invalid_token'
          }));
      }
    });
  }
});

app.get('/refresh_token', function (req, res) {

  // requesting access token from refresh token
  var refresh_token = req.query.refresh_token;
  var authOptions = {
    url: 'https://accounts.spotify.com/api/token',
    headers: { 'Authorization': 'Basic ' + (new Buffer.from(client_id + ':' + client_secret).toString('base64')) },
    form: {
      grant_type: 'refresh_token',
      refresh_token: refresh_token
    },
    json: true
  };

  request.post(authOptions, function (error, response, body) {
    if (!error && response.statusCode === 200) {
      access_token = body.access_token;
      res.send({
        'access_token': access_token
      });
    }
  });
});

app.get('/playlists', function (req, res) {
  console.log(access_token)
  var options = {
    url: 'https://api.spotify.com/v1/me/playlists',
    headers: { 'Authorization': 'Bearer ' + access_token },
    json: true
  };
  // use the access token to access the Spotify Web API
  request.get(options, function (error, response, body) {
    if (error) {
      console.log("error")
    }
    else {
      console.log("-----")

      items = response.body.items
      len = items.length
      playlists = []
      for (var i = 0; i < len; i++) {
        playlists.push(items[i].name)
      }
      console.log(playlists)
    }
  });
});

app.get('/most-played-tracks', function (req, res) {
  console.log(access_token)
  var options = {
    url: 'https://api.spotify.com/v1/me/top/tracks?limit=50',
    headers: { 'Authorization': 'Bearer ' + access_token },
    json: true
  };
  // use the access token to access the Spotify Web API
  request.get(options, function (error, response, body) {
    if (error) {
      console.log("error")
    }
    else {
      console.log("-----")
      items = response.body.items
      len = items.length
      songs = []
      ids = []
      for (var i = 0; i < len; i++) {
        songs.push(items[i].name)
        ids.push(items[i].id)
      }
      console.log(ids)

      for (var i = 0; i < len; i++) {
        var options = {
          url: 'https://api.spotify.com/v1/audio-features/' + ids[i],
          headers: { 'Authorization': 'Bearer ' + access_token },
          json: true
        };
        // use the access token to access the Spotify Web API
        request.get(options, function (error, response, body) {
          if (error) {
            console.log("error")
          }
          else {
            track = response.body
            feat = []
            feat.push(track.id)
            feat.push(track.danceability)
            feat.push(track.energy)
            feat.push(track.key)
            feat.push(track.loudness)
            feat.push(track.mode)
            feat.push(track.speechiness)
            feat.push(track.acousticness)
            feat.push(track.instrumentalness)
            feat.push(track.liveness)
            feat.push(track.valence)
            feat.push(track.tempo)
            // console.log(feat)
            // console.log(',')
          }
        });
      }
    }
  });
});


app.get('/most-played-artists', function (req, res) {
  console.log(access_token)
  var options = {
    url: 'https://api.spotify.com/v1/me/top/artists?limit=50',
    headers: { 'Authorization': 'Bearer ' + access_token },
    json: true
  };
  // use the access token to access the Spotify Web API
  request.get(options, function (error, response, body) {
    if (error) {
      console.log("error")
    }
    else {
      console.log("-----")
      items = response.body.items
      len = items.length
      artists = []
      for (var i = 0; i < len; i++) {
        artists.push(items[i].name)
      }
      console.log(artists)

      var ids = []
      for (var i = 0; i < len; i++) {
        ids.push(items[i].id)
      }
      console.log(ids)

      var no_of_artists = ids.length
      for (var i = 0; i < no_of_artists; i++) {
        var options = {
          url: 'https://api.spotify.com/v1/artists/' + ids[i] + '/albums?market=US',
          headers: { 'Authorization': 'Bearer ' + access_token },
          json: true
        };
        request.get(options, function (error, response, body) {
          if (error) {
            console.log("error")
          }
          else {
            // console.log("-----")
            items = response.body.items
            var arr = []
            var artist = items[0].artists[0].id
            arr.push(artist)
            var no_of_items = items.length
            var map = new Map()
            for (var i = 0; i < no_of_items; i++) {
              map.set(items[i].name, items[i].id)
            }
            const iterator = map.values()
            for (var i = 0; i < map.size; i++) {
              arr.push(iterator.next().value)
            }
            console.log(arr)
            console.log(',')
          }
        })
      }
    }
  });
});

app.get('/followed-artists', function (req, res) {
  console.log(access_token)
  var options = {
    url: 'https://api.spotify.com/v1/me/following?type=artist&limit=50',
    headers: { 'Authorization': 'Bearer ' + access_token },
    json: true
  };
  // use the access token to access the Spotify Web API
  request.get(options, function (error, response, body) {
    if (error) {
      console.log("error")
    }
    else {
      console.log("-----")
      items = response.body.artists.items
      len = items.length
      artists = []
      for (var i = 0; i < len; i++) {
        artists.push(items[i].name)
      }
      console.log(artists)

      var ids = []
      for (var i = 0; i < len; i++) {
        ids.push(items[i].id)
      }
      console.log(ids)

      var no_of_artists = ids.length
      for (var i = 0; i < no_of_artists; i++) {
        // console.log('[' + ids[i] + ", ")
        var options = {
          url: 'https://api.spotify.com/v1/artists/' + ids[i] + '/albums',
          headers: { 'Authorization': 'Bearer ' + access_token },
          json: true
        }
        request.get(options, function (error, response, body, options) {
          if (error) {
            console.log("error")
          }
          else {
            // var artist_id = options.url.substring(35)
            // var artist_id = artist_id.substring(0, artist_id.length - 7)
            // console.log('[' + options.url.substring(35) + ',')
            // console.log("-----")
            items = response.body.items
            var arr = []
            var artist = items[0].artists[0].id
            arr.push(artist)
            var no_of_items = items.length
            var map = new Map()
            for (var i = 0; i < no_of_items; i++) {
              map.set(items[i].name, items[i].id)
            }
            const iterator = map.values()
            for (var i = 0; i < map.size; i++) {
              arr.push(iterator.next().value)
            }
            console.log(arr)
            console.log(',')
          }
        })
      }
    }
  });
});

app.get('/saved-albums', function (req, res) {
  var options = {
    url: 'https://api.spotify.com/v1/me/albums?limit=50',
    headers: { 'Authorization': 'Bearer ' + access_token },
    json: true
  };
  // use the access token to access the Spotify Web API
  request.get(options, function (error, response, body) {
    if (error) {
      console.log("error")
    }
    else {
      console.log("-----")
      items = response.body.items
      len = items.length
      ids = []
      names = []
      for (var i = 0; i < len; i++) {
        ids.push(items[i].album.id)
        names.push(items[i].album.name)
      }
      console.log(names)
      console.log(ids)

      for (var i = 0; i < len; i++) {
        console.log(names[i])
        var options = {
          url: 'https://api.spotify.com/v1/albums/' + ids[i],
          headers: { 'Authorization': 'Bearer ' + access_token },
          json: true
        };
        console.log(ids[i])


        // use the access token to access the Spotify Web API
        request.get(options, function (error, response, body) {
          if (error) {
            console.log("error")
          }
          else {
            console.log("-----")
            items = response.body.tracks.items
            len = items.length
            songs = []
            songs.push(response.body.uri.substring(14,))
            for (var j = 0; j < len; j++) {
              songs.push(items[j].id)
            }
            console.log(songs)
          }
        });


      }

    }
  });
});

app.get('/get-artist-album', function (req, res) {
  id = req.query.id
  var options = {
    url: 'https://api.spotify.com/v1/tracks/' + id,
    headers: { 'Authorization': 'Bearer ' + access_token },
    json: true
  };
  // use the access token to access the Spotify Web API
  request.get(options, function (error, response, body) {
    if (error) {
      console.log("error")
    }
    else {
      console.log("-----")
      album = response.body.album.name
      console.log("album : ", album)
      artist = response.body.artists[0].name
      console.log("artist : ", artist)
    }
  });
});




app.get('/saved-tracks', function (req, res) {
  console.log(access_token)
  var options = {
    url: 'https://api.spotify.com/v1/me/tracks?limit=50',
    headers: { 'Authorization': 'Bearer ' + access_token },
    json: true
  };
  // use the access token to access the Spotify Web API
  request.get(options, function (error, response, body) {
    if (error) {
      console.log("error")
    }
    else {
      console.log("-----")
      items = response.body.items
      console.log(items)
      len = items.length
      ids = []
      songs = []
      for (var i = 0; i < len; i++) {
        ids.push(items[i].track.id)
        songs.push(items[i].track.name)
      }
      console.log(ids)
      console.log(songs)
      for (var i = 0; i < len; i++) {
        var options = {
          url: 'https://api.spotify.com/v1/audio-features/' + ids[i],
          headers: { 'Authorization': 'Bearer ' + access_token },
          json: true
        };
        // use the access token to access the Spotify Web API
        request.get(options, function (error, response, body) {
          if (error) {
            console.log("error")
          }
          else {
            track = response.body
            feat = []
            feat.push(track.id)
            feat.push(track.danceability)
            feat.push(track.energy)
            feat.push(track.key)
            feat.push(track.loudness)
            feat.push(track.mode)
            feat.push(track.speechiness)
            feat.push(track.acousticness)
            feat.push(track.instrumentalness)
            feat.push(track.liveness)
            feat.push(track.valence)
            feat.push(track.tempo)
            // console.log(feat)
            // console.log(',')
          }
        });
      }
    }
  });
});

// getting songs from ids final step
// app.get('/algorithm-rec', function (req, res) {
//   for (var i = 0; i < 1; i++) {
//     var options = {
//       url: 'https://api.spotify.com/v1/tracks?' + 'ids=3n3Ppam7vgaVa1iaRUc9Lp,3twNvmDtFQtAd5gMKedhLD',
//       headers: { 'Authorization': 'Bearer ' + access_token },
//       json: true
//     };
//     request.get(options, function (error, response, body) {
//       if (error) {
//         console.log("error")
//       }
//       else {
//         console.log(response.body.name)

//       }
//     })
//   }
// });

console.log('Listening on 8888');
app.listen(8888);