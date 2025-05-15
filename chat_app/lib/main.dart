import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'chat.dart'; // Import ChatScreen

void main() {
  runApp(MyApp());
}
const apiUrl = 'http://34.72.217.0:8000/generate';
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NyayaMitra',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),

      home: ChatScreen(apiUrl: apiUrl), // Pass the URL to ChatScreen
    );
  }
}

class UrlInputScreen extends StatefulWidget {
  const UrlInputScreen({super.key});

  @override
  _UrlInputScreenState createState() => _UrlInputScreenState();
}

class _UrlInputScreenState extends State<UrlInputScreen> {
  final TextEditingController _urlController = TextEditingController();
  String _response = '';

  Future<void> sendRequest() async {
    final String url = _urlController.text.trim();
    if (url.isEmpty) {
      setState(() {
        _response = 'Please enter a URL';
      });
      return;
    }

    try {
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        setState(() {
          _response = 'Connected! Navigating to chat...';
        });

        // Navigate to ChatScreen and pass the URL
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ChatScreen(apiUrl: url),
          ),
        );
      } else {
        setState(() {
          _response = 'Error: ${response.statusCode}';
        });
      }
    } catch (e) {
      setState(() {
        _response = 'Failed to connect: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Flutter to Flask')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _urlController,
              decoration: InputDecoration(
                labelText: 'Enter Flask URL',
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 10),
            ElevatedButton(
              onPressed: sendRequest,
              child: Text('Connect to Flask'),
            ),
            SizedBox(height: 20),
            Text(
              _response,
              style: TextStyle(fontSize: 16, color: Colors.black),
            ),
          ],
        ),
      ),
    );
  }
}



// Main App with yolo and other stuff


// import 'package:flutter/material.dart';
// import 'package:http/http.dart' as http;
// import 'connect.dart' as connect;

// void main() {
//   runApp(MyApp());
// }

// class MyApp extends StatelessWidget {
//   const MyApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       title: 'Flutter to Flask',
//       theme: ThemeData(
//         primarySwatch: Colors.blue,
//       ),
//       home: UrlInputScreen(),
//     );
//   }
// }

// class UrlInputScreen extends StatefulWidget {
//   const UrlInputScreen({super.key});

//   @override
//   _UrlInputScreenState createState() => _UrlInputScreenState();
// }

// class _UrlInputScreenState extends State<UrlInputScreen> {
//   final TextEditingController _urlController = TextEditingController();
//   String _response = '';

//   Future<void> sendRequest() async {
//     final String url = _urlController.text;
//     if (url.isEmpty) {
//       setState(() {
//         _response = 'Please enter a URL';
//       });
//       return;
//     }

//     try {
//       final response = await http.get(Uri.parse(url));
//       if (response.statusCode == 200) {
//         setState(() {
//           _response = response.body;
//         });
//         connect.main(url);
        
//       } 
//       else {
//         setState(() {
//           _response = 'Error: ${response.statusCode}';
//         });
//       }
//     } catch (e) {
//       setState(() {
//         _response = 'Failed to connect: $e';
//       });
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: Text('Flutter to Flask'),
//       ),
//       body: Padding(
//         padding: const EdgeInsets.all(16.0),
//         child: Column(
//           children: [
//             TextField(
//               controller: _urlController,
//               decoration: InputDecoration(
//                 labelText: 'Enter Flask URL',
//                 border: OutlineInputBorder(),
//               ),
//             ),
//             SizedBox(height: 10),
//             ElevatedButton(
//               onPressed: sendRequest,
//               child: Text('Connect to Flask'),
//             ),
//             SizedBox(height: 20),
//             Text(
//               _response,
//               style: TextStyle(fontSize: 16, color: Colors.black),
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }
